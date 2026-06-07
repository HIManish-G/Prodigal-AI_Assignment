"""
Identity verification logic.

Rules (from the assignment spec):
  - Full name MUST match exactly (case-sensitive, no fuzzy matching)
  - At least ONE secondary factor must also match:
      DOB (YYYY-MM-DD)  OR  Aadhaar last 4  OR  Pincode

This module is entirely deterministic Python — no LLM involved.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from state import ConversationState

log = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    passed: bool
    name_matched: bool
    secondary_matched: bool
    failure_reason: Optional[str] = None


def verify(state: ConversationState) -> VerificationResult:
    """
    Attempt verification given current state.
    Returns a VerificationResult (does NOT mutate state).
    """
    account = state.account_data
    if not account:
        return VerificationResult(
            passed=False,
            name_matched=False,
            secondary_matched=False,
            failure_reason="no_account_data",
        )

    # ── Name check (exact, case-sensitive) ───────────────────────────────────
    expected_name: str = account.get("full_name", "")
    provided_name: str = (state.provided_name or "").strip()
    name_ok = provided_name == expected_name

    # Enforce strict case-sensitivity by checking raw user message history
    # This prevents LLM auto-capitalization from bypassing case-sensitive checks
    if name_ok and len(state.history) > 0:
        user_inputs = [h["content"] for h in state.history if h["role"] == "user"]
        exact_match_found = any(expected_name in turn for turn in user_inputs)
        if not exact_match_found:
            log.info(
                "Case sensitivity violation: expected %r but exact casing not found in raw user inputs.",
                expected_name
            )
            name_ok = False

    if not name_ok:
        log.info(
            "Name mismatch: expected=%r  provided=%r",
            expected_name,
            provided_name,
        )
        return VerificationResult(
            passed=False,
            name_matched=False,
            secondary_matched=False,
            failure_reason="name_mismatch",
        )

    # ── Secondary factor ─────────────────────────────────────────────────────
    secondary_ok = False

    if state.provided_dob:
        if state.provided_dob == account.get("dob", ""):
            secondary_ok = True
            log.info("Secondary factor matched: dob")

    if not secondary_ok and state.provided_aadhaar:
        if state.provided_aadhaar == account.get("aadhaar_last4", ""):
            secondary_ok = True
            log.info("Secondary factor matched: aadhaar_last4")

    if not secondary_ok and state.provided_pincode:
        if state.provided_pincode == account.get("pincode", ""):
            secondary_ok = True
            log.info("Secondary factor matched: pincode")

    if not secondary_ok:
        log.info("No secondary factor matched.")
        return VerificationResult(
            passed=False,
            name_matched=True,
            secondary_matched=False,
            failure_reason="secondary_mismatch",
        )

    return VerificationResult(passed=True, name_matched=True, secondary_matched=True)