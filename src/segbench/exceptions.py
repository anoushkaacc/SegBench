"""Custom exceptions for SegBench."""


class SegBenchError(Exception):
    """Base exception for SegBench."""


class ValidationError(SegBenchError):
    """Raised when evaluation inputs are invalid."""
