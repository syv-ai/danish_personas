"""Domain errors raised by official-source services."""


class SourceServiceError(Exception):
    """Base class for failures in source services."""


class SourceAcquisitionError(SourceServiceError):
    """A source lock could not be resolved or fetched."""


class SourceArchiveError(SourceServiceError):
    """A raw-source archive could not be packed or restored."""
