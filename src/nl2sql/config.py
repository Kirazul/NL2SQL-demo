"""Settings, read once from `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Cloud zone. Models are named `provider/model`, and the three tiers are a
    # ladder the router climbs: a question is asked of the cheapest rung that can
    # answer it, and only moves up when that rung is rejected or unsure.
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    model_small: str = "openrouter/openai/gpt-4.1-nano"
    model_medium: str = "groq/openai/gpt-oss-20b"
    model_large: str = "groq/openai/gpt-oss-120b"
    model_fallback: str = "groq/qwen/qwen3.6-27b"

    cloud_timeout_s: float = 25.0
    cloud_max_retries: int = 2

    # Local zone
    local_gguf_path: Path | None = None
    local_gguf_repo: str = "unsloth/Qwen3-1.7B-GGUF"
    local_gguf_file: str = "Qwen3-1.7B-Q4_K_M.gguf"
    local_ctx: int = 8192
    local_threads: int = 4
    gliner_model: str = "models/gliner2-base-v1"
    gliner_fallback: str = "fastino/gliner2-base-v1"
    gliner_threshold: float = 0.45

    # Data
    db_path: Path = Field(default=DATA / "eicu.db")
    index_path: Path = Field(default=DATA / "index.db")
    sql_timeout_s: float = 15.0
    sql_max_rows: int = 2000

    # Security
    privacy_mode: Literal["strict", "demo"] = "demo"
    trace_dir: Path = Field(default=ROOT / "traces")

    # Observability
    langsmith_tracing: bool = True
    langsmith_api_key: str = ""
    langsmith_project: str = "nl2sql"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # Service
    api_host: str = "0.0.0.0"  # noqa: S104 — hosted runtimes need all interfaces
    api_port: int = 7860
    log_level: str = "INFO"

    @property
    def glossary_path(self) -> Path:
        return Path(__file__).parent / "resources" / "glossary.yaml"

    @property
    def allowlist_path(self) -> Path:
        return Path(__file__).parent / "resources" / "allowlist.yaml"

    @property
    def audit_log(self) -> Path:
        return self.trace_dir / "egress.jsonl"


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()
