"""
Conversation state — the single source of truth across all turns.
All fields are typed and optional; the stage enum drives control flow.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class Stage(str, Enum):
    # ── Onboarding ────────────────────────────────────────────────────────────
    INIT              = "INIT"               # First message; agent greets
    AWAIT_ACCOUNT_ID  = "AWAIT_ACCOUNT_ID"   # Waiting for / extracting account ID
    # ── Identity Verification ────────────────────────────────────────────────
    AWAIT_NAME        = "AWAIT_NAME"          # Need full name
    AWAIT_SECONDARY   = "AWAIT_SECONDARY"     # Need DOB / Aadhaar / Pincode
    # ── Payment ──────────────────────────────────────────────────────────────
    BALANCE_SHOWN     = "BALANCE_SHOWN"       # Balance shared; ask intent
    AWAIT_AMOUNT      = "AWAIT_AMOUNT"        # Collecting payment amount
    AWAIT_CARD        = "AWAIT_CARD"          # Progressive card collection
    CONFIRM_PAYMENT = "CONFIRM_PAYMENT"       # confirm payment before processing
    # ── Terminal ─────────────────────────────────────────────────────────────
    DONE              = "DONE"                # Successful flow complete
    LOCKED_OUT        = "LOCKED_OUT"          # Too many verification failures
    ERROR             = "ERROR"              # Unrecoverable error


@dataclass
class CardDetails:
    """Accumulates card fields across turns until all are present."""
    number: Optional[str]  = None   # 16-digit string, no spaces
    expiry_month: Optional[int]  = None
    expiry_year:  Optional[int]  = None
    cvv:           Optional[str]  = None
    cardholder_name: Optional[str] = None

    def missing_fields(self) -> List[str]:
        missing = []
        if not self.number:         missing.append("card number")
        if not self.expiry_month:   missing.append("expiry month")
        if not self.expiry_year:    missing.append("expiry year")
        if not self.cvv:            missing.append("CVV")
        if not self.cardholder_name: missing.append("cardholder name")
        return missing

    @property
    def complete(self) -> bool:
        return not self.missing_fields()


@dataclass
class ConversationState:
    """Full conversation state — everything the agent needs to resume any turn."""

    # ── Flow ──────────────────────────────────────────────────────────────────
    stage: Stage = Stage.INIT

    # ── Account ───────────────────────────────────────────────────────────────
    account_id:   Optional[str]  = None
    account_data: Optional[Dict[str, Any]] = None  # raw API response; never sent to user

    # ── Verification ─────────────────────────────────────────────────────────
    provided_name:    Optional[str] = None
    provided_dob:     Optional[str] = None   # normalised YYYY-MM-DD
    provided_aadhaar: Optional[str] = None   # 4 digits
    provided_pincode: Optional[str] = None   # 6 digits
    verified:         bool = False
    verify_attempts:  int  = 0

    # ── Payment ───────────────────────────────────────────────────────────────
    payment_amount:    Optional[float] = None
    card:              CardDetails = field(default_factory=CardDetails)
    card_attempts:     int = 0
    last_txn_id:       Optional[str] = None

    # ── Dialogue history (for LLM context) ───────────────────────────────────
    history: List[Dict[str, str]] = field(default_factory=list)

    # ── Helpers ───────────────────────────────────────────────────────────────
    @property
    def balance(self) -> Optional[float]:
        if self.account_data:
            return self.account_data.get("balance")
        return None

    @property
    def account_name(self) -> Optional[str]:
        """Name on the account (from API). Never expose to user directly."""
        if self.account_data:
            return self.account_data.get("full_name")
        return None

    def has_secondary_factor(self) -> bool:
        return bool(
            self.provided_dob or
            self.provided_aadhaar or
            self.provided_pincode
        )
