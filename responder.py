"""
Natural language response generator matching original signatures.
"""
from __future__ import annotations

import logging
import re
from typing import Optional, List, Dict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from config import OLLAMA_MODEL, OLLAMA_BASE_URL, LLM_RESPONSE_TEMP, LLM_REQUEST_TIMEOUT, CURRENCY_SYMBOL

log = logging.getLogger(__name__)

_SYSTEM = (
    "You are Niki, a warm, professional customer service agent for a financial service. "
    "You help users manage their accounts and complete authorized payments. "
    "Keep your responses friendly, polite, and concise (2-3 sentences)."
)

_KNOWLEDGE_BASE = """
[GENERAL SYSTEM KNOWLEDGE BASE]:
Strictly follow these system guidelines to assist and guide the customer if they are confused, ask questions, or provide incorrect details:

1. Account Identification:
   - Account ID: A unique identifier that always begins with the letters 'ACC' followed by 4-digits [ACCXXXX]. It can be found on welcome letters, invoices, or registration documents.

2. Identity Verification (Strict Indian Financial Standards):
   - Full Name: Must match their registered name exactly. (Matching is strictly case-sensitive; no fuzzy matching is allowed).
   - Date of Birth: Must be formatted as YYYY-MM-DD. Valid dates include standard leap year calendar dates (such as February 29).
   - Aadhaar ID: The last 4 digits of their Indian national identity card (exactly 4 consecutive digits).
   - Pincode: The 6-digit Indian postal code associated with their registered address.

3. Card Payment Details:
   - Card Number: Standard 15 or 16 consecutive digits.
   - Expiry Date: Expiry month (integer 1-12) and expiry year (4-digit format).
   - CVV: 3-digit standard security code (or 4-digit code for American Express) found on the card.
   - Cardholder Name: Accepted as-is to complete the transaction (it is not validated against the account holder's name).

4. Partial Payments:
   - Partial payments (paying less than the total outstanding balance) are fully supported.
"""

_llm: Optional[ChatOllama] = None


def _get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_RESPONSE_TEMP,
            timeout=LLM_REQUEST_TIMEOUT,
        )
    return _llm


def _ask(instruction: str, fallback: str) -> str:
    try:
        llm = _get_llm()
        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=instruction)
        ]
        resp = llm.invoke(messages)
        text = resp.content.strip()
        
        # Clean up any accidental chat-template formatting prefixes
        text = re.sub(r"^assistant\s*", "", text, flags=re.IGNORECASE).strip()
        return text if text else fallback
    except Exception as exc:
        log.warning("Responder LLM error: %s — using fallback", exc)
        return fallback


# ── Stage-specific generators (Aligned to original positional parameters) ─────

def greet(history: Optional[List[Dict[str, str]]] = None) -> str:
    return _ask(
        "Greet the user warmly and ask for their account ID to get started.",
        "Hello! Welcome to payment services. Could you please share your 7-character account ID to get started?"
    )


def ask_account_id(retry: bool = False, history: Optional[List[Dict[str, str]]] = None) -> str:
    if retry:
        return _ask(
            "The account ID wasn't found. Politely ask the customer to check and re-enter their Account ID (e.g. ACCXXXX).",
            "I couldn't find that account. Could you double-check your account ID? It should look like ACCXXXX."
        )
    return _ask(
        "Politely ask the customer to share their Account ID to get started.",
        "Could you please share your account ID? It should start with 'ACC' followed by digits, like ACCXXXX."
    )


def ask_name(history: Optional[List[Dict[str, str]]] = None) -> str:
    return _ask(
        "Ask the customer to confirm their full name so we can securely verify their account.",
        "To verify your identity, could you please confirm your full name as it appears on your account?"
    )


def ask_secondary_factor(name_provided: bool = True, history: Optional[List[Dict[str, str]]] = None) -> str:
    return _ask(
        "Ask the customer to provide either their date of birth, the last 4 digits of their AADHAR ID, or their pincode to verify their identity.",
        "To complete verification, could you please provide either your date of birth, the last 4 digits of your Aadhaar, or your pincode?"
    )


def verification_failed_retry(attempts_left: int, history: Optional[List[Dict[str, str]]] = None) -> str:
    return _ask(
        f"The verification attempt failed. The user has {attempts_left} attempt(s) remaining. Ask them to try again with their full name plus one of: Date of Birth, the last 4 digits of Aadhaar, or their pincode.",
        f"The details didn't match our records. Please try again with your full name and one of: date of birth, Aadhaar last 4 digits, or pincode."
    )


def verification_locked_out(history: Optional[List[Dict[str, str]]] = None) -> str:
    return _ask(
        "The user has exhausted all verification attempts. Inform them the account is locked for security and close professionally.",
        "Too many failed verification attempts. For security, we've locked the session. Please contact customer support."
    )


def show_balance(balance: float, name: str, volunteered_amount: Optional[float] = None, history: Optional[List[Dict[str, str]]] = None) -> str:
    formatted_balance = f"{CURRENCY_SYMBOL}{balance:,.2f}"
    if volunteered_amount is not None:
        formatted_amt = f"{CURRENCY_SYMBOL}{volunteered_amount:,.2f}"
        instruction = (
            f"The customer's identity is verified. Greet them by name ({name}) and tell them their balance is {formatted_balance}. "
            f"Confirm that you will proceed with their requested payment of {formatted_amt}, and ask them to provide their card details to complete the transaction."
        )
    else:
        instruction = f"The customer's identity is verified. Greet them by name ({name}) and tell them their balance is {formatted_balance}. Ask how much they want to pay today."
        
    return _ask(instruction, f"Identity verified! Hello, {name}. Your outstanding balance is {formatted_balance}. How much would you like to pay?")


def ask_payment_amount(balance: float, history: Optional[List[Dict[str, str]]] = None) -> str:
    formatted = f"{CURRENCY_SYMBOL}{balance:,.2f}"
    return _ask(
        f"Ask how much they'd like to pay against their outstanding balance of {formatted}.",
        f"How much would you like to pay? Your balance is {formatted}."
    )


def invalid_amount(reason: str, balance: float, history: Optional[List[Dict[str, str]]] = None) -> str:
    formatted = f"{CURRENCY_SYMBOL}{balance:,.2f}"
    return _ask(
        f"The payment amount is invalid: {reason}. The outstanding balance is {formatted}. Ask for a valid amount.",
        f"That amount isn't valid. Please enter an amount up to {formatted}."
    )


def ask_card_details(missing_fields: list, history: Optional[List[Dict[str, str]]] = None) -> str:
    fields_str = ", ".join(missing_fields)
    return _ask(
        f"Ask the user to provide the missing card details: {fields_str}.",
        f"Please provide your card details — I still need: {fields_str}."
    )


def card_validation_error(error: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    return _ask(
        f"The card details are invalid: {error}. Ask them to check and re-enter.",
        f"There's an issue with your card details: {error}. Could you please check and re-enter them?"
    )


def payment_success(txn_id: str, amount: float, remaining: Optional[float] = None, history: Optional[List[Dict[str, str]]] = None) -> str:
    formatted_amount = f"{CURRENCY_SYMBOL}{amount:,.2f}"
    remaining_str = f" Your remaining balance is {CURRENCY_SYMBOL}{remaining:,.2f}." if remaining is not None else ""
    return _ask(
        f"The payment of {formatted_amount} was processed successfully! Inform the user of the successful payment and provide their Transaction ID: {txn_id}. Thank them and close warmly.",
        f"Payment successful! Transaction ID: {txn_id}.{remaining_str}"
    )


def payment_error(error_code: str, amount: float, retryable: bool, history: Optional[List[Dict[str, str]]] = None) -> str:
    retry_note = "Please check details and try again." if retryable else "Please contact support."
    return _ask(
        f"Payment failed due to {error_code}. Inform them clearly and politely.",
        f"Payment failed. {retry_note}"
    )


def closing(history: Optional[List[Dict[str, str]]] = None) -> str:
    return _ask("Give a brief, warm closing message.", "Thanks for reaching out. Have a wonderful day!")


# ── Context-Aware Conversational Router ───────────────────────────────────────

def generate_chitchat(history: List[Dict[str, str]], metadata_prompt: str) -> str:
    """Generates warm, human replies to casual chat without repeating rigid prompts."""
    try:
        llm = _get_llm()
        messages = [SystemMessage(content=_SYSTEM + "\n" + _KNOWLEDGE_BASE)]
        for turn in history[-4:]:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))
                
        prompt = (
            f"{metadata_prompt}\n"
            f"[BEHAVIORAL DIRECTIVES]:\n"
            f"1. HIGH-PRIORITY: Genuinely help, answer, and converse with the user. If they ask where to find required information, or comment on verification (like SSNs), use [NIKI'S PRE-PREPARED Q&A KNOWLEDGE BASE] to answer their question in detail with warm support.\n"
            f"2. LOW-PRIORITY: The collection of [NEXT NEEDED DATA] is a low-priority background task on this turn. Only soft-steer them back to it at the very end of your response if it feels 100% natural, otherwise focus entirely on helping them first. Do not sound robotic or pushy. Keep it to 2-3 sentences."
        )
        messages.append(SystemMessage(content=prompt))
        resp = llm.invoke(messages)
        
        text = resp.content.strip()
        text = re.sub(r"^assistant\s*", "", text, flags=re.IGNORECASE).strip()
        return text
    except Exception as exc:
        log.warning("Chitchat responder error: %s", exc)
        return "Hello! I am here to assist with your payment. Let me know when you are ready to get started."