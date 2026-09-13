"""Temporal validation helpers that fail closed against future leakage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedSplit:
    """Indexes for a chronological split with a label-overlap purge."""

    train_index: pd.Index
    test_index: pd.Index
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    purge_days: int


def _as_normalized_dates(values: Iterable[object]) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(list(values), utc=True)).normalize().tz_localize(None)
    if dates.hasnans:
        raise ValueError("Decision dates must not contain missing values")
    return dates


def purged_chronological_split(
    frame: pd.DataFrame,
    *,
    test_start: object,
    purge_days: int = 30,
    date_column: str = "decision_date",
) -> PurgedSplit:
    """Split by a shared date boundary and purge overlapping future labels."""
    if purge_days < 0:
        raise ValueError("purge_days must be non-negative")
    if date_column not in frame:
        raise ValueError(f"Missing date column: {date_column}")

    dates = _as_normalized_dates(frame[date_column])
    boundary = pd.Timestamp(test_start)
    if boundary.tzinfo is not None:
        boundary = boundary.tz_convert("UTC").tz_localize(None)
    boundary = boundary.normalize()
    train_end = boundary - pd.Timedelta(days=purge_days + 1)

    train_mask = dates <= train_end
    test_mask = dates >= boundary
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Both train and test partitions must contain rows")

    return PurgedSplit(
        train_index=frame.index[train_mask],
        test_index=frame.index[test_mask],
        train_end=train_end,
        test_start=boundary,
        purge_days=purge_days,
    )


def backward_asof_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: str,
    by: str | Sequence[str] | None = None,
    tolerance: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Join observations available at or before each decision timestamp.

    ``nearest`` is deliberately not exposed. The source timestamp is retained
    as ``{on}_source`` so callers can audit that no future row was selected.
    """
    if on not in left or on not in right:
        raise ValueError(f"Both frames must contain {on}")
    source_column = f"{on}_source"
    if source_column in right:
        raise ValueError(f"Reserved column already exists: {source_column}")

    left_sorted = left.copy()
    right_sorted = right.copy()
    left_sorted[on] = pd.to_datetime(left_sorted[on], utc=True)
    right_sorted[on] = pd.to_datetime(right_sorted[on], utc=True)
    right_sorted[source_column] = right_sorted[on]

    sort_columns = [on] + ([by] if isinstance(by, str) else list(by or []))
    left_sorted = left_sorted.sort_values(sort_columns)
    right_sorted = right_sorted.sort_values(sort_columns)
    joined = pd.merge_asof(
        left_sorted,
        right_sorted,
        on=on,
        by=by,
        direction="backward",
        allow_exact_matches=True,
        tolerance=tolerance,
    )
    future = joined[source_column].notna() & (joined[source_column] > joined[on])
    if future.any():
        raise AssertionError("A future observation was selected by the as-of join")
    return joined


class TrainOnlyPreprocessor:
    """Select and standardize features using training rows only."""

    def __init__(self) -> None:
        self.feature_names_: list[str] = []
        self.means_: pd.Series | None = None
        self.scales_: pd.Series | None = None
        self.fit_start_: pd.Timestamp | None = None
        self.fit_end_: pd.Timestamp | None = None
        self.fit_rows_: int = 0

    def fit(
        self,
        train: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        date_column: str = "decision_date",
    ) -> "TrainOnlyPreprocessor":
        if not feature_columns:
            raise ValueError("At least one feature column is required")
        missing = [column for column in feature_columns if column not in train]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        numeric = train.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        usable = [
            column
            for column in feature_columns
            if numeric[column].notna().any() and float(numeric[column].std(ddof=0)) > 0.0
        ]
        if not usable:
            raise ValueError("No finite, non-constant training feature is available")

        self.feature_names_ = usable
        self.means_ = numeric[usable].mean()
        scales = numeric[usable].std(ddof=0)
        self.scales_ = scales.where(scales > 0.0, 1.0)
        self.fit_rows_ = int(len(train))
        if date_column in train:
            dates = _as_normalized_dates(train[date_column])
            self.fit_start_ = dates.min()
            self.fit_end_ = dates.max()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.means_ is None or self.scales_ is None:
            raise RuntimeError("The preprocessor must be fitted before transform")
        missing = [column for column in self.feature_names_ if column not in frame]
        if missing:
            raise ValueError(f"Missing fitted feature columns: {missing}")
        numeric = frame.loc[:, self.feature_names_].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        return (numeric - self.means_) / self.scales_

    def fit_transform(
        self,
        train: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        date_column: str = "decision_date",
    ) -> pd.DataFrame:
        return self.fit(train, feature_columns, date_column=date_column).transform(train)

    def metadata(self) -> dict[str, object]:
        if self.means_ is None or self.scales_ is None:
            raise RuntimeError("The preprocessor has not been fitted")
        return {
            "feature_names": list(self.feature_names_),
            "means": {key: float(value) for key, value in self.means_.items()},
            "scales": {key: float(value) for key, value in self.scales_.items()},
            "fit_start": (
                self.fit_start_.date().isoformat() if self.fit_start_ is not None else None
            ),
            "fit_end": self.fit_end_.date().isoformat() if self.fit_end_ is not None else None,
            "fit_rows": self.fit_rows_,
            "selection_policy": "finite_non_constant_on_training_only",
        }
