"""Explicit holding horizons, distinct from historical signal windows."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Horizon:
    label: str
    holding_sessions_min: int
    holding_sessions_max: int
    signal_sessions: int
    history_calendar_days: int

    def metadata(self):
        return {**asdict(self), "method": "historical_screen", "forward_validated": False,
                "validation_status": "No validated forward-return forecast"}


RECOMMENDATION_HORIZONS = {
    "short": Horizon("1-2 Weeks", 5, 10, 10, 90),
    "medium": Horizon("1 Month", 21, 21, 21, 120),
    "long": Horizon("3-6 Months", 63, 126, 63, 220),
}
OPPORTUNITY_HORIZONS = {
    "short": Horizon("1-3 Months", 21, 63, 42, 120),
    "medium": Horizon("6-12 Months", 126, 252, 189, 365),
    "long": Horizon("2-3 Years", 504, 756, 756, 1200),
}
