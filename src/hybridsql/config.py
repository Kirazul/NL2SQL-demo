"""Configuration read from `.env`. Single source of settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Cloud zone (outside the trust boundary) ----------------------------------
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    groq_model_fallback: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-oss-20b:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    cloud_provider_chain: str = "groq,openrouter"
    cloud_timeout_s: float = 25.0
    cloud_max_retries: int = 2

    # --- Local zone (inside the trust boundary) -----------------------------------
    local_llm_backend: Literal["llamacpp", "ollama", "hf-inference"] = "llamacpp"
    local_llm_gguf_repo: str = "unsloth/Qwen3-1.7B-GGUF"
    local_llm_gguf_file: str = "Qwen3-1.7B-Q4_K_M.gguf"
    # Path to a GGUF already on disk. When set, it short-circuits the download:
    # `huggingface_hub` cannot resume an interrupted transfer (its temp filename
    # derives from the signed CDN URL, which changes on every attempt), which is
    # unworkable for a 1 GB file.
    local_llm_gguf_path: Path | None = None
    local_llm_ctx: int = 8192
    local_llm_threads: int = 4

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:1.7b"

    hf_token: str = ""
    hf_inference_model: str = "Qwen/Qwen3-4B-Instruct-2507"

    gliner_model: str = "models/gliner2-base-v1"
    gliner_fallback_model: str = "fastino/gliner2-base-v1"
    gliner_threshold: float = 0.45

    # --- Database -----------------------------------------------------------------
    db_path: Path = Field(default=ROOT / "data" / "warehouse" / "eicu.db")
    db_readonly: bool = True
    sql_timeout_s: float = 10.0
    sql_max_rows: int = 2000

    # --- Security -----------------------------------------------------------------
    privacy_mode: Literal["strict", "demo"] = "demo"
    egress_allowlist_path: Path = Field(default=ROOT / "config" / "allowlist.yaml")
    egress_audit_log: Path = Field(default=ROOT / "traces" / "egress.jsonl")

    # --- Observability -------------------------------------------------------------
    trace_dir: Path = Field(default=ROOT / "traces")


    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "unimed-hybridsql"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # --- Kaggle (publishing the demo notebook) ---------------------------------------
    kaggle_username: str = ""
    kaggle_api_token: str = ""

    # --- Application ----------------------------------------------------------------
    api_host: str = "0.0.0.0"  # noqa: S104 — hosted runtimes require all interfaces
    api_port: int = 7860
    log_level: str = "INFO"

    # --- Derived paths ---------------------------------------------------------------
    @property
    def value_index_path(self) -> Path:
        return self.db_path.with_name("value_index.db")

    @property
    def glossary_path(self) -> Path:
        return ROOT / "config" / "glossary.yaml"

    @property
    def provider_chain(self) -> list[str]:
        return [p.strip() for p in self.cloud_provider_chain.split(",") if p.strip()]


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()
