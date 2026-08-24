"""Model access: one module for the cloud, one for the local model."""

from nl2sql.llm import cloud, local

__all__ = ["cloud", "local"]
