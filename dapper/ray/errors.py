"""Errors raised while starting and validating a Dapper Ray cluster."""


class RayBootstrapError(RuntimeError):
    """Raised when a Ray node cannot be started or proven ready."""
