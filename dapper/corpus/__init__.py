"""Storage-agnostic corpus access.

Everything in the dedup and curriculum pipelines addresses data by URI, which
may be local or ``gs://``. This package holds the one layer that knows the
difference, so nothing above it has to.
"""
