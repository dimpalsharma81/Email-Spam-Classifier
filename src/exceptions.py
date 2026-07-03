"""Custom exceptions for the spam classifier."""


class SpamClassifierError(Exception):
    """Base exception for spam classifier."""
    pass


class DataProcessingError(SpamClassifierError):
    """Raised when data processing fails."""
    pass


class FeatureExtractionError(SpamClassifierError):
    """Raised when feature extraction fails."""
    pass


class ModelError(SpamClassifierError):
    """Raised when model operation fails."""
    pass


class PredictionError(SpamClassifierError):
    """Raised when prediction fails."""
    pass
