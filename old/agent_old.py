"""
Payment Collection AI Agent
============================
Implements the required interface:

    class Agent:
        def next(self, user_input: str) -> dict:
            ...

Architecture
------------
Declarative checklist state machine. On every turn, the agent gathers any
discoverable entities from the input and prompts for the first incomplete
item on the data checklist.
"""
from __future__ import annotations

import logging
from typing import Optional

from config import MAX_VERIFICATION_ATTEMPTS, MAX_CARD_ATTEMPTS, CURRENCY_SYMBOL
from state import ConversationState, Stage, CardDetails
import extractor
import responder
from tools import lookup_account, process_payment
from verifier import verify

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


class Agent:
    """Payment collection agent. Maintains all state between next() calls."""

    def __init__(self) -> None:
        self.state = ConversationState()
        # Ephemeral slot to hold an extracted amount during pre-flight checklist validation
        self._temp_extracted_amount: Optional[float] = None

    # ── Public interface ──────────────────────────────────────────────────────

    def next(self, user_input: str) -> dict:
        """
        Process one turn of the conversation.
        """
        user_input = (user_input or "").strip()
        log.info("TURN  stage=%-20s input=%r", self.state.stage, user_input)

        self.state.history.append({"role": "user", "content": user_input})

        # 1. ── Pre-Extraction Checklist Sweep (Asynchronous Gatherer) ──
        self._sweep_checklist_inputs(user_input)

        # 2. ── Run Checklist Validation Flow ──
        response = self._run_checklist_flow(user_input)

        self.state.history.append({"role": "assistant", "content": response})
        log.info("REPLY stage=%-20s reply=%r", self.state.stage, response)

        return {"message": response}

    # ── Checklist Operations ──────────────────────────────────────────────────

    def next(self, user_input: str) -> dict:
        """Process one turn using Dynamic Extraction Control (Gen 8 style)."""
        user_input = (user_input or "").strip()
        log.info("TURN  stage=%-20s input=%r", self.state.stage, user_input[:80])

        # 1. Deterministically extract & redact sensitive data in Python
        # (This populates card number, DOB, Pincode, and CVV before the LLM gets the text)
        sanitized_input = extractor.redact_and_extract(user_input, self.state.stage.value, self.state)

        # 2. Append sanitized text to history (keeps LLM blind to raw card/DOB data)
        self.state.history.append({"role": "user", "content": sanitized_input})

        # 3. Execute the cascading dynamic checklist flow
        response = self._run_dynamic_checklist(sanitized_input)

        self.state.history.append({"role": "assistant", "content": response})
        return {"message": response}

    def _run_dynamic_checklist(self, text: str) -> str:
        """Cascading Step-by-Step Checklist Engine."""
        # Loop up to 5 times in a single turn to allow mid-turn state cascades
        for _ in range(5):
            
            # ── Step 1: Account ID ──
            if not self.state.account_id:
                acc_id = extractor.extract_account_id(text)
                if acc_id:
                    self.state.account_id = acc_id
                    continue  # State changed! Recalculate turn order (cascade loop)
                else:
                    self.state.stage = Stage.AWAIT_ACCOUNT_ID
                    return responder.ask_account_id(self.state.history)

            # ── Step 2: Account Lookup ──
            if self.state.account_id and not self.state.account_data:
                log.info("Looking up account: %s", self.state.account_id)
                result = lookup_account(self.state.account_id)
                if not result["success"]:
                    self.state.account_id = None
                    return responder.ask_account_id(self.state.history, retry=True)
                self.state.account_data = result["data"]
                
                if self.state.balance == 0.0:
                    self.state.stage = Stage.DONE
                    return "Your account has a zero balance. There is nothing to pay, and we have closed this session. Thank you!"
                continue  # State changed! Cascade loop

            # ── Step 3: Identity Verification ──
            if not self.state.verified:
                # Confirm name
                if not self.state.provided_name:
                    name = extractor.extract_full_name(text)
                    if name:
                        self.state.provided_name = name
                        continue  # Name populated! Cascade loop
                    else:
                        self.state.stage = Stage.AWAIT_NAME
                        return responder.ask_name(self.state.history)

                # Confirm secondary verification factor
                if not self.state.has_secondary_factor():
                    secondary = extractor.extract_secondary_factor(text)
                    self._absorb_secondary(secondary)
                    if self.state.has_secondary_factor():
                        continue  # Secondary factor populated! Cascade loop
                    else:
                        self.state.stage = Stage.AWAIT_SECONDARY
                        return responder.ask_secondary_factor(self.state.history)

                # Strict Verification Check
                self.state.verify_attempts += 1
                verify_result = verify(self.state)
                log.info("Verification attempt #%d: passed=%s", self.state.verify_attempts, verify_result.passed)

                if verify_result.passed:
                    self.state.verified = True
                    self.state.stage = Stage.AWAIT_AMOUNT
                    
                    # Check if they volunteered the payment amount early on this exact turn
                    amt = extractor.extract_amount(text, balance=self.state.balance)
                    if amt is not None:
                        validation_error = self._get_amount_validation_error(amt)
                        if not validation_error:
                            self.state.payment_amount = amt
                    
                    # Step 4 (PDF): Share the outstanding balance
                    return responder.show_balance(
                        self.state.history, 
                        self.state.balance, 
                        self.state.account_name,
                        volunteered_amount=self.state.payment_amount
                    )
                else:
                    remaining = MAX_VERIFICATION_ATTEMPTS - self.state.verify_attempts
                    if remaining <= 0:
                        self.state.stage = Stage.LOCKED_OUT
                        return responder.verification_locked_out(self.state.history)

                    # Clear invalid details to force clean retry
                    self.state.provided_name = None
                    self.state.provided_dob = None
                    self.state.provided_aadhaar = None
                    self.state.provided_pincode = None
                    self.state.stage = Stage.AWAIT_SECONDARY
                    return responder.verification_failed_retry(self.state.history, remaining)

            # ── Step 4: Payment Amount ──
            # If we haven't completed card collection, we allow them to overwrite/correct the amount
            if self.state.payment_amount is None or not self.state.card.complete:
                # CVV Guard: If we already have the card number and they input a 3-4 digit CVV,
                # bypass amount extraction to prevent the CVV from overriding their payment amount.
                is_cvv_input = (self.state.card.number is not None) and bool(re.match(r"^\s*\d{3,4}\s*$", text))
                
                if not is_cvv_input:
                    amt = extractor.extract_amount(text, balance=self.state.balance)
                    if amt is not None:
                        validation_error = self._get_amount_validation_error(amt)
                        if not validation_error:
                            if self.state.payment_amount != amt:
                                self.state.payment_amount = amt
                                log.info("Overwrote payment_amount with corrected value: %s", amt)
                                continue  # Amount updated! Cascade loop
                        else:
                            return responder.invalid_amount(self.state.history, validation_error, self.state.balance)
                
                if self.state.payment_amount is None:
                    self.state.stage = Stage.AWAIT_AMOUNT
                    return responder.ask_payment_amount(self.state.history, self.state.balance)

            # ── Step 5: Card Details Collection & Payment Processing ──
            if not self.state.card.cardholder_name:
                self.state.card.cardholder_name = self.state.account_name

            missing = self.state.card.missing_fields()
            if missing:
                self.state.stage = Stage.AWAIT_CARD
                return responder.ask_card_details(self.state.history, missing)

            # Validate card formats strictly before triggering API processing
            validation_error = extractor.validate_card(self.state.card)
            if validation_error:
                self.state.card_attempts += 1
                if self.state.card_attempts >= MAX_CARD_ATTEMPTS:
                    self.state.stage = Stage.ERROR
                    return "Too many invalid card attempts. Please verify your details and try again."

                if "card number" in validation_error: self.state.card.number = None
                elif "expiry" in validation_error or "expired" in validation_error:
                    self.state.card.expiry_month = None
                    self.state.card.expiry_year = None
                elif "CVV" in validation_error: self.state.card.cvv = None

                self.state.stage = Stage.AWAIT_CARD
                return responder.card_validation_error(self.state.history, validation_error)

            return self._process_payment()

        return responder.closing(self.state.history)

    def _run_checklist_flow(self, text: str) -> str:
        """Step-by-step validation engine based on information completeness."""
        # ── Step 1: Account Identification ──
        if not self.state.account_id:
            self.state.stage = Stage.AWAIT_ACCOUNT_ID
            return responder.ask_account_id()

        if self.state.account_id and not self.state.account_data:
            log.info("Looking up account: %s", self.state.account_id)
            result = lookup_account(self.state.account_id)
            if not result["success"]:
                # Clear invalid ID so user can re-provide it
                self.state.account_id = None
                return responder.ask_account_id(retry=True)
            self.state.account_data = result["data"]
            log.info("Account loaded: id=%s balance=%s", self.state.account_id, self.state.balance)

        # ── Step 2: Zero Balance [Fixed Outcome — No LLM Call] ──
        if self.state.account_data and self.state.balance == 0.0:
            self.state.stage = Stage.DONE
            return "Your account has a zero balance. There is nothing to pay, and we have closed this session. Thank you!"

        # ── Step 3: Identity Verification ──
        if self.state.account_data and not self.state.verified:
            # Check for missing full name
            if not self.state.provided_name:
                self.state.stage = Stage.AWAIT_NAME
                return responder.ask_name()

            # Check for missing secondary verification factor
            if not self.state.has_secondary_factor():
                self.state.stage = Stage.AWAIT_SECONDARY
                return responder.ask_secondary_factor(name_provided=True)

            # Both verification inputs are now present — execute strict validation
            self.state.verify_attempts += 1
            verify_result = verify(self.state)
            log.info(
                "Verification attempt #%d: passed=%s name=%s secondary=%s",
                self.state.verify_attempts,
                verify_result.passed,
                verify_result.name_matched,
                verify_result.secondary_matched,
            )

            if verify_result.passed:
                self.state.verified = True
                self.state.stage = Stage.BALANCE_SHOWN
                return responder.show_balance(self.state.balance, self.state.account_name)

            # Handle mismatch failures & retry boundaries
            remaining = MAX_VERIFICATION_ATTEMPTS - self.state.verify_attempts
            if remaining <= 0:
                self.state.stage = Stage.LOCKED_OUT
                return responder.verification_locked_out()

            # Reset both checklist factors completely to force a clean retry sequence
            self.state.provided_name    = None
            self.state.provided_dob     = None
            self.state.provided_aadhaar = None
            self.state.provided_pincode = None

            self.state.stage = Stage.AWAIT_SECONDARY
            return responder.verification_failed_retry(remaining)

        # ── Step 4: Payment Amount ──
        if self.state.verified and self.state.payment_amount is None:
            amt = getattr(self, "_temp_extracted_amount", None)
            if amt is not None:
                validation_error = self._get_amount_validation_error(amt)
                if validation_error:
                    return responder.invalid_amount(validation_error, self.state.balance)
                self.state.payment_amount = amt
                # Note: DO NOT return yet. We fall through to evaluate Step 5 on this same turn!
            else:
                self.state.stage = Stage.AWAIT_AMOUNT
                return responder.ask_payment_amount(self.state.balance)

        # ── Step 5: Card Details Collection & Payment Processing ──
        if self.state.verified and self.state.payment_amount is not None:
            if not self.state.card.cardholder_name:
                self.state.card.cardholder_name = self.state.account_name

            missing = self.state.card.missing_fields()
            if missing:
                self.state.stage = Stage.AWAIT_CARD
                return responder.ask_card_details(missing)

            # Validate card formats strictly before triggering API processing
            validation_error = extractor.validate_card(self.state.card)
            if validation_error:
                self.state.card_attempts += 1
                if self.state.card_attempts >= MAX_CARD_ATTEMPTS:
                    self.state.stage = Stage.ERROR
                    return "Too many invalid card attempts. Please verify your details and try again."

                # Reset only the specific invalid card field
                if "card number" in validation_error:
                    self.state.card.number = None
                elif "expiry" in validation_error or "expired" in validation_error:
                    self.state.card.expiry_month = None
                    self.state.card.expiry_year  = None
                elif "CVV" in validation_error:
                    self.state.card.cvv = None
                elif "cardholder name" in validation_error:
                    self.state.card.cardholder_name = None

                self.state.stage = Stage.AWAIT_CARD
                return responder.card_validation_error(validation_error)

            return self._process_payment()

        return self._handle_terminal(text)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _absorb_secondary(self, factors: dict) -> None:
        if "dob" in factors and not self.state.provided_dob:
            self.state.provided_dob = factors["dob"]
        if "aadhaar_last4" in factors and not self.state.provided_aadhaar:
            self.state.provided_aadhaar = factors["aadhaar_last4"]
        if "pincode" in factors and not self.state.provided_pincode:
            self.state.provided_pincode = factors["pincode"]

    def _get_amount_validation_error(self, amount: float) -> Optional[str]:
        balance = self.state.balance
        if amount <= 0:
            return "amount must be greater than zero"
        if amount > balance:
            return f"amount exceeds your balance of {CURRENCY_SYMBOL}{balance:,.2f}"
        return None

    def _process_payment(self) -> str:
        card = self.state.card
        log.info(
            "Processing payment: account=%s amount=%s card=****%s",
            self.state.account_id,
            self.state.payment_amount,
            card.number[-4:] if card.number else "????",
        )

        result = process_payment(
            account_id=self.state.account_id,
            amount=self.state.payment_amount,
            card_number=card.number,
            cvv=card.cvv,
            expiry_month=card.expiry_month,
            expiry_year=card.expiry_year,
            cardholder_name=card.cardholder_name,
        )

        if result["success"]:
            txn_id = result["transaction_id"]
            self.state.last_txn_id = txn_id
            self.state.stage = Stage.DONE

            remaining = self.state.balance - self.state.payment_amount
            resp = responder.payment_success(
                txn_id=txn_id,
                amount=self.state.payment_amount,
                remaining=remaining if remaining > 0 else None,
            )

            # Pre-emptively append the transaction reference ID if missing from LLM response
            if "transaction" not in resp.lower() and "txn_" not in resp.lower():
                resp += f"\n\nTransaction Reference ID: {txn_id}"
            return resp

        error_code = result.get("error_code", "api_error")
        log.info("Payment failed: error_code=%s", error_code)

        RETRYABLE_ERRORS = {"invalid_card", "invalid_cvv", "invalid_expiry", "invalid_amount"}
        retryable = error_code in RETRYABLE_ERRORS

        self.state.card_attempts += 1

        if retryable and self.state.card_attempts < MAX_CARD_ATTEMPTS:
            if error_code == "invalid_card":
                self.state.card.number = None
            elif error_code == "invalid_cvv":
                self.state.card.cvv = None
            elif error_code == "invalid_expiry":
                self.state.card.expiry_month = None
                self.state.card.expiry_year  = None
            elif error_code == "invalid_amount":
                self.state.payment_amount = None
                self.state.stage = Stage.AWAIT_AMOUNT

            return responder.payment_error(error_code, self.state.payment_amount or 0, retryable=True)

        self.state.stage = Stage.ERROR
        return responder.payment_error(error_code, self.state.payment_amount or 0, retryable=False)

    def _handle_terminal(self, text: str) -> str:
        if self.state.stage == Stage.DONE:
            return responder.closing()
        if self.state.stage == Stage.LOCKED_OUT:
            return (
                "This session has been locked due to too many verification failures. "
                "Please contact customer support for assistance."
            )
        return (
            "This session has ended. "
            "Please contact customer support or start a new session."
        )