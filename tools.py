"""
Thin, typed wrappers around the external Payment Collection API.
Each function returns a Result dict with `success` bool and either `data` or `error`.
No business logic lives here — only I/O.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, Optional

import requests

from config import API_BASE_URL, API_TIMEOUT

log = logging.getLogger(__name__)

# ── Result type alias ─────────────────────────────────────────────────────────

Result = Dict[str, Any]


def _post(endpoint: str, payload: dict) -> Result:
    """Generic POST with robust error classification, exponential backoff, and jitter."""
    url = f"{API_BASE_URL}{endpoint}"
    max_retries = 3
    backoff_base = 1.0 
    
    # Transient HTTP status codes that are safe to retry
    TRANSIENT_STATUS_CODES = {502, 503, 504}
    
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=API_TIMEOUT)
            log.debug("POST %s  status=%d (attempt %d/%d)", url, resp.status_code, attempt, max_retries)
            
            # If it's a temporary gateway overload, back off and retry
            if resp.status_code in TRANSIENT_STATUS_CODES:
                if attempt < max_retries:
                    sleep_time = (backoff_base * (2 ** (attempt - 1))) + random.uniform(0.1, 0.5)
                    log.warning("Transient HTTP %d on %s. Retrying in %.2fs...", resp.status_code, url, sleep_time)
                    time.sleep(sleep_time)
                    continue
            
            return {
                "http_status": resp.status_code,
                "body": resp.json() if resp.content else {},
            }
            
        except (requests.Timeout, requests.ConnectionError) as exc:
            # Timeouts and DNS resolution errors are transient and safe to retry
            if attempt < max_retries:
                sleep_time = (backoff_base * (2 ** (attempt - 1))) + random.uniform(0.1, 0.5)
                log.warning("Transient network error (%s) on %s. Retrying in %.2fs...", type(exc).__name__, url, sleep_time)
                time.sleep(sleep_time)
                continue
            
            # Out of retries; fail gracefully
            err_type = "timeout" if isinstance(exc, requests.Timeout) else "connection_error"
            log.error("API call failed after %d attempts: %s", max_retries, exc)
            return {"http_status": None, "body": {}, "network_error": err_type}
            
        except Exception as exc:
            # Terminal exceptions (payload bugs, unhandled crashes) - do not retry
            log.exception("Terminal exception calling %s: %s", url, exc)
            return {"http_status": None, "body": {}, "network_error": str(exc)}


# ── Public API ────────────────────────────────────────────────────────────────

def lookup_account(account_id: str) -> Result:
    """
    POST /api/lookup-account
    Returns:
        success=True  + data dict on 200
        success=False + error_code on 404
        success=False + error_code="api_error" on anything else
    """
    raw = _post("/api/lookup-account", {"account_id": account_id})

    if raw.get("network_error"):
        return {"success": False, "error_code": "network_error",
                "message": f"Network issue: {raw['network_error']}"}

    status = raw["http_status"]
    body   = raw["body"]

    if status == 200:
        return {"success": True, "data": body}
    if status == 404:
        return {"success": False, "error_code": body.get("error_code", "account_not_found"),
                "message": body.get("message", "Account not found.")}
    # unexpected
    return {"success": False, "error_code": "api_error",
            "message": f"Unexpected API response (HTTP {status})."}


def process_payment(
    account_id: str,
    amount: float,
    card_number: str,
    cvv: str,
    expiry_month: int,
    expiry_year: int,
    cardholder_name: str,
) -> Result:
    """
    POST /api/process-payment
    Returns:
        success=True  + transaction_id on 200
        success=False + error_code on 422
        success=False + error_code="api_error" on anything else
    """
    payload = {
        "account_id": account_id,
        "amount": round(amount, 2),
        "payment_method": {
            "type": "card",
            "card": {
                "cardholder_name": cardholder_name,
                "card_number": card_number,
                "cvv": cvv,
                "expiry_month": expiry_month,
                "expiry_year": expiry_year,
            },
        },
    }

    raw = _post("/api/process-payment", payload)

    if raw.get("network_error"):
        return {"success": False, "error_code": "network_error",
                "message": f"Network issue: {raw['network_error']}"}

    status = raw["http_status"]
    body   = raw["body"]

    if status == 200 and body.get("success"):
        return {"success": True, "transaction_id": body.get("transaction_id", "N/A")}
    if status == 422 or (status == 200 and not body.get("success")):
        error_code = body.get("error_code", "payment_failed")
        return {"success": False, "error_code": error_code,
                "message": body.get("message", "")}
    return {"success": False, "error_code": "api_error",
            "message": f"Unexpected API response (HTTP {status})."}
