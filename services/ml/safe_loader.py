"""
Safe ML Model Loading Utilities

Security module for loading ML models (pickle, PyTorch) with path validation
to prevent arbitrary code execution from untrusted sources.

Security Measures:
- Path traversal protection (validates paths are within SAFE_MODEL_DIR)
- PyTorch weights_only=True by default (fallback if needed)
- Comprehensive logging for audit trail

Usage:
    from services.ml.safe_loader import safe_pickle_load, safe_torch_load

    # Instead of: model = pickle.load(f)
    model = safe_pickle_load(model_path)

    # Instead of: checkpoint = torch.load(model_file)
    checkpoint = safe_torch_load(model_file, map_location='cpu')

Author: SmartFolio Security Team
Date: 2025-11-22
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Safe directory for ML models (all models must be within this directory)
SAFE_MODEL_DIR = Path("cache/ml_pipeline")


class UnsafeModelPathError(ValueError):
    """Raised when attempting to load model from unsafe path"""
    pass


def safe_pickle_load(file_path: str | Path) -> Any:
    """
    Safely load pickled ML model with path validation

    Security: Only loads from SAFE_MODEL_DIR to prevent arbitrary code execution
    from untrusted pickle files.

    Args:
        file_path: Path to pickle file (must be within SAFE_MODEL_DIR)

    Returns:
        Unpickled model object

    Raises:
        UnsafeModelPathError: If path is outside SAFE_MODEL_DIR
        FileNotFoundError: If model file doesn't exist

    Example:
        >>> model = safe_pickle_load("cache/ml_pipeline/models/regime/model.pkl")
    """
    abs_path = Path(file_path).resolve()
    safe_dir = SAFE_MODEL_DIR.resolve()

    # Path traversal protection
    try:
        abs_path.relative_to(safe_dir)
    except ValueError:
        raise UnsafeModelPathError(
            f"Unsafe model path (outside {safe_dir}): {file_path}\n"
            f"Resolved to: {abs_path}\n"
            f"All ML models must be within the safe directory."
        )

    if not abs_path.exists():
        raise FileNotFoundError(f"Model file not found: {abs_path}")

    logger.info(f"Loading pickle model from validated path: {abs_path}")

    try:
        with open(abs_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug(f"Successfully loaded pickle model: {abs_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load pickle model {abs_path}: {e}")
        raise


def safe_torch_load(
    file_path: str | Path,
    map_location: Optional[str] = 'cpu',
    weights_only: Optional[bool] = None
) -> Any:
    """
    Safely load PyTorch model with path validation

    Security:
    - Path traversal protection (validates path within SAFE_MODEL_DIR)
    - Attempts weights_only=True first (PyTorch 2.0+ security best practice)
    - Falls back to weights_only=False if needed for custom layers

    Args:
        file_path: Path to PyTorch .pth/.pt file (must be within SAFE_MODEL_DIR)
        map_location: Device to map tensors to (default: 'cpu')
        weights_only: If True, only load weights (safer). If None, auto-detect.

    Returns:
        PyTorch model or state dict

    Raises:
        UnsafeModelPathError: If path is outside SAFE_MODEL_DIR
        FileNotFoundError: If model file doesn't exist

    Example:
        >>> checkpoint = safe_torch_load("cache/ml_pipeline/models/regime_neural.pth")
        >>> model.load_state_dict(checkpoint)
    """
    # Import torch here to avoid dependency if not needed
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch not installed. Install with: pip install torch")

    abs_path = Path(file_path).resolve()
    safe_dir = SAFE_MODEL_DIR.resolve()

    # Path traversal protection
    try:
        abs_path.relative_to(safe_dir)
    except ValueError:
        raise UnsafeModelPathError(
            f"Unsafe model path (outside {safe_dir}): {file_path}\n"
            f"Resolved to: {abs_path}\n"
            f"All ML models must be within the safe directory."
        )

    if not abs_path.exists():
        raise FileNotFoundError(f"Model file not found: {abs_path}")

    # Auto-detect: try weights_only=True first, fallback if needed
    if weights_only is None:
        try:
            logger.info(f"Loading PyTorch model (weights_only=True): {abs_path}")
            model = torch.load(abs_path, map_location=map_location, weights_only=True)
            logger.debug(f"Successfully loaded with weights_only=True: {abs_path}")
            return model
        except Exception as e:
            logger.warning(
                f"Model {abs_path.name} requires weights_only=False (custom layers): {e}"
            )
            logger.info(f"Loading PyTorch model (weights_only=False): {abs_path}")
            model = torch.load(abs_path, map_location=map_location, weights_only=False)
            logger.debug(f"Successfully loaded with weights_only=False: {abs_path}")
            return model
    else:
        # Explicit weights_only parameter provided
        logger.info(f"Loading PyTorch model (weights_only={weights_only}): {abs_path}")
        model = torch.load(abs_path, map_location=map_location, weights_only=weights_only)
        logger.debug(f"Successfully loaded PyTorch model: {abs_path}")
        return model


def validate_model_path(file_path: str | Path) -> Path:
    """
    Validate that a path is safe for model storage/loading

    Args:
        file_path: Path to validate

    Returns:
        Validated absolute Path object

    Raises:
        UnsafeModelPathError: If path is outside SAFE_MODEL_DIR
    """
    abs_path = Path(file_path).resolve()
    safe_dir = SAFE_MODEL_DIR.resolve()

    try:
        abs_path.relative_to(safe_dir)
    except ValueError:
        raise UnsafeModelPathError(
            f"Unsafe model path (outside {safe_dir}): {file_path}\n"
            f"Resolved to: {abs_path}"
        )

    return abs_path


# Export public API
__all__ = [
    'safe_pickle_load',
    'safe_torch_load',
    'validate_model_path',
    'UnsafeModelPathError',
    'SAFE_MODEL_DIR',
]
