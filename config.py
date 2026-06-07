"""
Central configuration for the Payment Collection Agent.
Edit OLLAMA_MODEL to swap models — any Ollama model with tool/JSON support works.
Recommended for 4070 laptop (8 GB VRAM):
  - llama3.1:8b   (~4.5 GB VRAM, excellent tool-calling)   ← default
  - qwen2.5:7b    (~4.5 GB VRAM, strong instruction follow)
  - mistral-nemo  (~7.5 GB VRAM, bigger context, higher quality)
"""
import os
def get_require_confirmation() -> bool:
    return os.getenv("REQUIRE_PAYMENT_CONFIRMATION", "false").lower() == "true"
# ── LLM ──────────────────────────────────────────────────────────────────────
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Generation parameters tuned for local inference
LLM_TEMPERATURE = 0.0        # deterministic extraction
LLM_RESPONSE_TEMP = 0.3      # slightly warmer for natural responses
LLM_REQUEST_TIMEOUT = 120    # seconds; generous for first-token latency

# ── External API ──────────────────────────────────────────────────────────────
API_BASE_URL    = os.getenv(
    "PAYMENT_API_BASE_URL",
    "https://se-payment-verification-api.service.external.usea2.aws.prodigaltech.com"
)
API_TIMEOUT     = 30   # seconds per HTTP request

# ── Business Logic ────────────────────────────────────────────────────────────
MAX_VERIFICATION_ATTEMPTS = 3   # lock-out after this many failures
MAX_CARD_ATTEMPTS         = 3   # retries for card/payment errors

# ── Currency ──────────────────────────────────────────────────────────────────
CURRENCY_SYMBOL = "₹"
