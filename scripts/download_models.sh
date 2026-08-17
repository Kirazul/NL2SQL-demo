#!/usr/bin/env bash
# =============================================================================
#  Download the local models — with reliable resume
# =============================================================================
#  Why not use `huggingface_hub`?
#  Because the name of its `.incomplete` file derives from the signed CDN URL,
#  which changes on every attempt. On an unstable connection each retry therefore
#  restarts from zero — observed here: 504 MB lost twice in a row.
#
#  `curl -C -` resumes at the real byte, in a file with a stable name. The models
#  land in `models/`, outside the HF cache, and are loaded by local path. No
#  network is needed afterwards.
#
#      bash scripts/download_models.sh
# =============================================================================
set -uo pipefail

BASE="https://huggingface.co"
DEST="${DEST:-models}"

# --- GLiNER2: entity extraction (stage 1) ------------------------------------
GLINER_REPO="fastino/gliner2-base-v1"
GLINER_FILES=(
  config.json
  model.safetensors
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  added_tokens.json
  spm.model
  # Essential and easy to miss: `from_pretrained` looks for the DeBERTa encoder
  # config in this subdirectory. Without it, loading fails with a misleading
  # message ("Repo id must use alphanumeric chars"), because transformers takes
  # the missing path for a repository identifier.
  encoder_config/config.json
)

# --- Qwen3-1.7B: answer writing (stage 3) ------------------------------------
# The `unsloth` repository rather than `Qwen`: the official one only publishes
# Q8_0 (1.7 GB). Q4_K_M, half the size for equivalent quality on this writing
# task, exists only in the community repository.
GGUF_REPO="unsloth/Qwen3-1.7B-GGUF"
GGUF_FILE="Qwen3-1.7B-Q4_K_M.gguf"

fetch() {
  local repo="$1" file="$2" directory="$3"
  local target="$directory/$file"
  local url="$BASE/$repo/resolve/main/$file"
  mkdir -p "$(dirname "$target")"

  # Already here and complete? Do not download it again.
  #
  # Without this check, `curl -C -` on an already-complete file receives a 416
  # (range not satisfiable), fails because of -f, and the script "retries" five
  # times before reporting a failure for a perfectly valid file. On Kaggle, where
  # setup is re-run on every saved version, this saves 1.9 GB of transfer — or
  # five false failures.
  if [ -f "$target" ]; then
    local remote local_size
    remote=$(curl -fsIL "$url" | tr -d '\r' \
             | awk 'tolower($1) == "content-length:" { size = $2 } END { print size }')
    local_size=$(wc -c < "$target" | tr -d ' ')
    if [ -n "$remote" ] && [ "$remote" = "$local_size" ]; then
      printf '  %-34s already present (%s)\n' "$file" "$(du -h "$target" | cut -f1)"
      return 0
    fi
  fi

  # -C - : resume. --retry : CDN interruptions are common at these sizes.
  for attempt in 1 2 3 4 5; do
    printf '  %-34s attempt %d\n' "$file" "$attempt"
    if curl -fL --progress-bar -C - \
            --retry 5 --retry-delay 3 --retry-all-errors \
            --connect-timeout 30 \
            -o "$target" "$url"; then
      printf '  %-34s OK (%s)\n' "$file" "$(du -h "$target" | cut -f1)"
      return 0
    fi
    sleep 5
  done
  printf '  %-34s FAILED after 5 attempts\n' "$file"
  return 1
}

echo "== GLiNER2 ($GLINER_REPO) =="
for f in "${GLINER_FILES[@]}"; do
  fetch "$GLINER_REPO" "$f" "$DEST/gliner2-base-v1"
done

echo
echo "== Qwen3-1.7B GGUF ($GGUF_REPO) =="
fetch "$GGUF_REPO" "$GGUF_FILE" "$DEST/qwen3-1.7b"

echo
echo "== Result =="
du -sh "$DEST"/* 2>/dev/null || true
echo
echo "Then set in .env:"
echo "  GLINER_MODEL=models/gliner2-base-v1"
echo "  LOCAL_LLM_GGUF_PATH=models/qwen3-1.7b/$GGUF_FILE"
