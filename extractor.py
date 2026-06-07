"""
NLP extraction layer — converts free-form user text into structured values.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Optional, Any

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from config import OLLAMA_MODEL, OLLAMA_BASE_URL, LLM_TEMPERATURE, LLM_REQUEST_TIMEOUT
from state import CardDetails

log = logging.getLogger(__name__)

_llm: Optional[ChatOllama] = None


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
            format="json", 
            timeout=LLM_REQUEST_TIMEOUT,
        )
    return _llm


def _llm_json(system_prompt: str, user_text: str) -> dict:
    """Ask the LLM for a single JSON object. Returns {} on any failure."""
    try:
        llm = get_llm()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_text),
        ]
        response = llm.invoke(messages)
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
        return json.loads(raw)
    except Exception as exc:
        log.warning("LLM extraction error: %s", exc)
        return {}


# ── Digits as Words Mapper ────────────────────────────────────────────────────

_DIGIT_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
}


# ── Account ID ────────────────────────────────────────────────────────────────

_ACC_RE = re.compile(r"\bACC\s*[-]?\s*(\d+)\b", re.IGNORECASE)

def extract_account_id(text: str) -> Optional[str]:
    """Extract an account ID like ACC1001 from free-form text."""
    # Fast regex path
    matches = _ACC_RE.findall(text)
    if matches:
        return f"ACC{matches[-1]}"

    # LLM fallback
    result = _llm_json(
        system_prompt=(
            "You extract account IDs from user messages. "
            "Account IDs always start with the letters 'ACC' followed by 4-digits [ACCXXXX]. "
            "Return ONLY a JSON object with key 'account_id' (string or null). "
            "Do not include any explanation."
        ),
        user_text=text,
    )
    raw_id = result.get("account_id")
    if raw_id:
        raw_id_str = str(raw_id).strip()
        # Python 1-to-1 exact check on user text
        if raw_id_str in text:
            m2 = _ACC_RE.search(raw_id_str)
            if m2:
                return f"ACC{m2.group(1)}"
    return None


# ── Full Name ─────────────────────────────────────────────────────────────────
_STOP_WORDS: frozenset = frozenset({
    # pronouns / determiners
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    # verbs
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "will", "would", "shall", "should", "may", "might", "must",
    "can", "could", "ought",
    # conjunctions / prepositions / articles
    "a", "an", "the", "and", "but", "if", "or", "nor", "so", "yet",
    "as", "at", "by", "for", "in", "of", "on", "to", "up", "with",
    "about", "above", "after", "against", "before", "below", "between",
    "during", "from", "into", "out", "over", "through", "under",
    # adverbs / misc
    "not", "no", "just", "very", "also", "too", "than", "then",
    "here", "there", "when", "where", "why", "how", "all", "both",
    "only", "own", "same", "few", "more", "most", "other", "such",
    "again", "once", "further",
    # contractions (apostrophe stripped before lookup)
    "im", "ive", "id", "youre", "youve", "youd", "youll",
    "hes", "shes", "weve", "theyre", "theyve", "dont", "doesnt",
    "didnt", "cant", "wont", "wouldnt", "shouldnt", "couldnt",
    "isnt", "arent", "wasnt", "werent", "hasnt", "havent", "hadnt",
    # domain words safe to strip in this context
    "name", "full", "first", "last", "call", "called", "known",
    "hi", "hello", "hey", "yes", "yeah", "yep", "nope",
    "okay", "ok", "sure", "please", "thanks", "thank",
    "account", "number", "born",
})

_NAME_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z'\-]*$")
_NEVER_NAME_RE = re.compile(r"^(?:ACC\d+|\d+)$", re.IGNORECASE)


# ── Helper Functions ──────────────────────────────────────────────────────────

def _strip_stopwords_preserve_caps(text: str) -> str:
    caps = [ch.isupper() for ch in text]
    lowered = text.lower()
    kept = []
    for m in re.finditer(r"\S+", lowered):
        token = m.group()
        start = m.start()
        lookup_key = re.sub(r"[^a-z]", "", token)  # letters only for stop-word check
        if lookup_key in _STOP_WORDS:
            continue
        restored = "".join(
            ch.upper() if caps[start + i] else ch
            for i, ch in enumerate(token)
        )
        kept.append(restored)
    return " ".join(kept)


def _is_plausible_name(candidate: str) -> bool:
    tokens = candidate.strip().split()
    if len(tokens) < 2 or len(tokens) > 6:
        return False
    return all(
        bool(_NAME_TOKEN_RE.match(t)) and not bool(_NEVER_NAME_RE.match(t))
        for t in tokens
    )


def _get_kept_words_and_indices(text: str) -> list[tuple[str, int]]:
    """
    Tokenise by whitespace, strip stop-words, and return a list of
    tuples: (kept_word_with_original_casing, original_word_index).
    """
    raw_words = text.strip().split()
    kept = []
    for i, word in enumerate(raw_words):
        # Strip punctuation (like the comma in "Hi,") for stop-word check
        lookup_key = re.sub(r"[^a-z]", "", word.lower())
        if lookup_key in _STOP_WORDS:
            continue
        kept.append((word, i))
    return kept


def extract_full_name(text: str) -> Optional[str]:
    # Run stop-word cleaning to build the stripped string for fast-path evaluation
    stripped = _strip_stopwords_preserve_caps(text)

    # 1. ── Fast-Path Evaluation ──
    if _is_plausible_name(stripped):
        tokens = stripped.split()
        
        # We allow contiguous name blocks between 2 and 4 words
        if 2 <= len(tokens) <= 4:
            raw_words = text.strip().split()
            
            # Map the exact consecutive word positions of the kept tokens in the original text
            indices = []
            last_idx = 0
            is_contiguous = True
            for token in tokens:
                # Find the token in raw_words starting from last_idx
                clean_token = re.sub(r"[^a-zA-Z]", "", token.lower())
                found = False
                for i in range(last_idx, len(raw_words)):
                    clean_word = re.sub(r"[^a-zA-Z]", "", raw_words[i].lower())
                    if clean_token == clean_word:
                        indices.append(i)
                        last_idx = i + 1
                        found = True
                        break
                if not found:
                    is_contiguous = False
                    break
            
            if is_contiguous:
                # Ensure the matched indices are strictly adjacent (difference of 1)
                if len(indices) == len(tokens):
                    is_contiguous = all(indices[i] - indices[i-1] == 1 for i in range(1, len(indices)))
                else:
                    is_contiguous = False

            if is_contiguous:
                # Ensure every token on the fast-path starts with an uppercase letter
                if all(t[0].isupper() for t in tokens):
                    return stripped

    # 2. ── LLM Fallback Evaluation (For complex clauses or lowercase inputs) ──
    # We send the raw, completely untouched text to the LLM to preserve all semantic context
    result = _llm_json(
        system_prompt=(
            "You are going to extract a person's full name (containing both the first name and the last name) from a message. "
            "Disregard conversational introductory wrappers or self-identification phrases. "
            "Return ONLY a JSON object with key 'full_name' containing the extracted full name as a string, "
            "or null if no full name is mentioned. "
            "Preserve original spelling exactly."
        ),
        user_text=text,
    )

    name = result.get("full_name")
    if not name:
        return None

    name_str = str(name).strip()

    if name_str.lower() in {"null", "none", "unknown", "n/a", "na", "nil", "undefined", ""}:
        return None

    # LLM-extracted full legal name plausibility check
    if not _is_plausible_name(name_str):
        log.info("Rejected implausible LLM name %r from input %r", name_str, text[:80])
        return None

    # Python 1-to-1 exact casing substring restore
    idx = text.lower().find(name_str.lower())
    if idx != -1:
        return text[idx: idx + len(name_str)]

    return name_str

# ── Date of Birth ─────────────────────────────────────────────────────────────

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}

def _parse_dob_string(raw: str) -> Optional[str]:
    raw = raw.strip()

    # YYYY-MM-DD
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", raw)
    if m:
        return _validate_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # DD-MM-YYYY or DD/MM/YYYY
    m = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$", raw)
    if m:
        return _validate_date(int(m.group(3)), int(m.group(2)), int(m.group(1)))

    # DD MonthName YYYY or MonthName DD, YYYY
    m = re.match(
        r"^(\d{1,2})\s+([A-Za-z]+)\s+(\d{2,4})$|"
        r"^([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{2,4})$", raw)
    if m:
        if m.group(1):
            day, month_str, year = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        else:
            month_str, day, year = m.group(4).lower(), int(m.group(5)), int(m.group(6))
        month = _MONTHS.get(month_str[:3])
        if month:
            if year < 100:
                year += 1900
            return _validate_date(year, month, day)

    return None


def _validate_date(year: int, month: int, day: int) -> Optional[str]:
    """Deterministically validate the date constraints in Python."""
    try:
        dt = datetime(year, month, day)
        # Python check: Ensure a Date of Birth is not in the future
        if dt > datetime.today():
            log.info("Rejected date %04d-%02d-%02d: Date is in the future.", year, month, day)
            return None
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        # Python natively catches invalid days, months, and leap year anomalies
        log.info("Rejected date %04d-%02d-%02d: Calendar logically invalid.", year, month, day)
        return None


def extract_amount(text: str, balance: Optional[float] = None) -> Optional[float]:
    """
    Extract a numeric payment amount (in INR) from the user message.
    """
    balance_hint = f"The current balance is {balance}." if balance is not None else ""
    result = _llm_json(
        system_prompt=(
            f"Extract the numeric payment amount in INR that the user wishes to pay. {balance_hint} "
            "If the user's message semantically expresses a desire to pay the total outstanding sum, "
            "fully clear the account, or settle the entire balance, resolve this intent "
            "to the numerical balance amount provided. "
            "Return ONLY a JSON object: {\"amount\": <number_or_null>}. "
            "Do not include any other text."
        ),
        user_text=text,
    )
    raw = result.get("amount")
    if raw is not None:
        try:
            val = round(float(raw), 2)
            if val > 0:
                if re.search(r"\d", text):
                    val_str = str(int(val))
                    clean_text = text.replace(",", "")
                    if val_str in clean_text:
                        return val
                else:

                    return val
        except (ValueError, TypeError):
            pass
    return None


# ── Aadhaar Last 4 ────────────────────────────────────────────────────────────

def extract_aadhaar_last4(text: str) -> Optional[str]:
    result = _llm_json(
        system_prompt=(
            "Extract the last 4 digits of an Aadhaar number from the user message. "
            "It must consist of exactly 4 consecutive digits. "
            "Return ONLY a JSON object with key 'aadhaar_last4' containing the digits, or null if not found."
        ),
        user_text=text,
    )
    raw = result.get("aadhaar_last4")
    if raw:
        raw_str = str(raw).strip()
        # Python 1-to-1 exact substring check on raw user text
        if raw_str in text:
            if re.match(r"^\d{4}$", raw_str):
                return raw_str
    return None


# ── Pincode ───────────────────────────────────────────────────────────────────

def extract_pincode(text: str) -> Optional[str]:
    result = _llm_json(
        system_prompt=(
            "Extract a 6-digit Indian postal pincode from the user message. "
            "It must consist of exactly 6 digits, possibly spaced out. "
            "Return ONLY a JSON object with key 'pincode' containing the 6 digits, or null if not found."
        ),
        user_text=text,
    )
    raw = result.get("pincode")
    if raw:
        pincode = str(raw).strip()
        # Python 1-to-1 exact substring check on raw user text
        if pincode in text:
            if re.match(r"^\d{6}$", pincode):
                return pincode
    return None


# ── Secondary Factor (multi-type extraction) ──────────────────────────────────

def extract_secondary_factor(text: str) -> dict:
    out = {}

    # 1. Deterministic Regex Pre-extraction (Bypasses LLM on raw digits)
    # Standalone 4-digit number -> Aadhaar Last 4
    if re.match(r"^\s*\d{4}\s*$", text):
        out["aadhaar_last4"] = text.strip()

    # Spaced or Standard 6-digit number -> Pincode
    pin_matches = re.findall(r"\b(?:\d\s*?){6}\b", text)
    if pin_matches:
        clean_pin = re.sub(r"\s+", "", pin_matches[-1])
        out["pincode"] = clean_pin

    # DOB standard patterns (YYYY-MM-DD or DD/MM/YYYY)
    parsed_dob = _parse_dob_string(text)
    if parsed_dob:
        out["dob"] = parsed_dob

    # 2. LLM Fallback Extraction
    result = _llm_json(
        system_prompt=(
            "Extract identity verification fields from the user message. "
            "Fields to look for:\n"
            "  - dob: date of birth in YYYY-MM-DD format\n"
            "  - aadhaar_last4: last 4 digits of Aadhaar (exactly 4 digits)\n"
            "  - pincode: 6-digit postal code\n"
            "Return ONLY a JSON object with any of these keys that are present. "
            "Set missing fields to null."
        ),
        user_text=text,
    )

    # Merge LLM results with strict 1-to-1 validation
    dob_raw = result.get("dob")
    if not out.get("dob") and dob_raw:
        dob_str = str(dob_raw).strip()
        if dob_str.lower() not in ("null", "none", "unknown", "n/a", "na", "nil", "undefined", ""):
            parsed = _parse_dob_string(dob_str)
            if parsed:
                # Python 1-to-1 exact substring check on the DOB year
                year = parsed.split("-")[0]
                if year in text:
                    out["dob"] = parsed

    aadhaar = result.get("aadhaar_last4")
    if not out.get("aadhaar_last4") and aadhaar:
        aadhaar_str = str(aadhaar).strip()
        if aadhaar_str.lower() not in ("null", "none", "unknown", "n/a", "na", "nil", "undefined", ""):
            if re.match(r"^\d{4}$", aadhaar_str):
                # Python 1-to-1 exact substring check
                if aadhaar_str in text:
                    out["aadhaar_last4"] = aadhaar_str

    pincode = result.get("pincode")
    if not out.get("pincode") and pincode:
        pincode_str = str(pincode).strip()
        if pincode_str.lower() not in ("null", "none", "unknown", "n/a", "na", "nil", "undefined", ""):
            if re.match(r"^\d{6}$", pincode_str):
                # Python 1-to-1 exact substring check
                if pincode_str in text:
                    out["pincode"] = pincode_str

    return out


# ── Card Details ──────────────────────────────────────────────────────────────

def _luhn_check(number: str) -> bool:
    digits = [int(d) for d in reversed(number)]
    total = 0
    for i, d in enumerate(digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def extract_card_details(text: str, existing: Optional[CardDetails] = None) -> CardDetails:
    card = existing or CardDetails()

    # Pre-clean word-digits (e.g. "one two three") to numeric digits
    clean_text = text.lower()
    for word, digit in _DIGIT_WORDS.items():
        clean_text = re.sub(rf"\b{word}\b", digit, clean_text)

    # Remove all spaces for clean digit matching
    no_spaces_text = re.sub(r"\s+", "", clean_text)

    # 1. Deterministic Regex Pre-extraction
    # Card Number
    num_match = re.search(r"\b(?:\d[ -]*?){15,16}\b", text)
    if num_match:
        clean_num = re.sub(r"\D", "", num_match.group(0))
        if len(clean_num) in (15, 16):
            card.number = clean_num

    # Expiry MM/YY or MM/YYYY
    expiry_match = re.search(r"\b(\d{1,2})\s*/\s*(\d{2,4})\b", text)
    if expiry_match:
        try:
            m = int(expiry_match.group(1))
            y = int(expiry_match.group(2))
            if 1 <= m <= 12:
                card.expiry_month = m
                if y < 100:
                    y += 2000
                if 2020 <= y <= 2060:
                    card.expiry_year = y
        except ValueError:
            pass

    # Expiry Month name strings (checked against clean text to support converted digits)
    for month_name, m_val in _MONTHS.items():
        if re.search(rf"\b{month_name}\b", clean_text, re.I):
            card.expiry_month = m_val
            year_match = re.search(r"\b(20\d{2}|\d{2})\b", clean_text)
            if year_match:
                try:
                    y = int(year_match.group(1))
                    if y < 100:
                        y += 2000
                    if 2020 <= y <= 2060:
                        card.expiry_year = y
                except ValueError:
                    pass
            break

    # Standalone CVV: exactly 3 or 4 digits
    if re.match(r"^\s*\d{3,4}\s*$", no_spaces_text):
        card.cvv = no_spaces_text

    # CVV preceded by CVV label (including spaced digits like "cvv 1 2 3")
    cvv_match = re.search(r"\bcvv\b\s*(?:is|:)?\s*((?:\d\s*?){3,4})\b", clean_text, re.I)
    if cvv_match:
        card.cvv = re.sub(r"\s", "", cvv_match.group(1))

    # 2. LLM Parser Fallback
    result = _llm_json(
        system_prompt=(
            "Extract payment card details from the user message.\n"
            "Fields to look for:\n"
            "  card_number: digits only, no spaces\n"
            "  expiry_month: integer 1-12\n"
            "  expiry_year: 4-digit integer\n"
            "  cvv: string of 3 or 4 digits\n"
            "  cardholder_name: full name as written on card\n"
            "Return ONLY JSON with any subset of these keys that are present. Set missing fields to null."
        ),
        user_text=text,
    )

    if not card.number and result.get("card_number"):
        num_str = str(result["card_number"]).strip()
        if re.sub(r"\D", "", num_str) in no_spaces_text:
            clean_num = re.sub(r"\D", "", num_str)
            if len(clean_num) in (15, 16):
                card.number = clean_num

    if not card.expiry_month and result.get("expiry_month") is not None:
        try:
            m = int(result["expiry_month"])
            if str(m) in no_spaces_text:
                if 1 <= m <= 12:
                    card.expiry_month = m
        except (ValueError, TypeError):
            pass

    if not card.expiry_year and result.get("expiry_year") is not None:
        try:
            y = int(result["expiry_year"])
            y_str = str(y)[-2:] if y > 100 else str(y)
            if y_str in no_spaces_text:
                if y < 100:
                    y += 2000
                if 2020 <= y <= 2060:
                    card.expiry_year = y
        except (ValueError, TypeError):
            pass

    if not card.cvv and result.get("cvv") is not None:
        cvv_str = str(result["cvv"]).strip()
        if cvv_str in no_spaces_text:
            clean_cvv = re.sub(r"\D", "", cvv_str)
            if len(clean_cvv) in (3, 4):
                card.cvv = clean_cvv

    if not card.cardholder_name and result.get("cardholder_name"):
        name = str(result["cardholder_name"]).strip()
        if name.lower() not in ("null", "none", "unknown", "n/a", "na", "nil", "undefined", ""):
            if name in text:
                if len(name) >= 2:
                    card.cardholder_name = name

    return card


def validate_card(card: CardDetails) -> Optional[str]:
    """Validate card details deterministically in Python."""
    if not card.number:
        return "card number is missing"
    if not re.match(r"^\d{15,16}$", card.number):
        return "card number must be 15 or 16 digits"
    if not _luhn_check(card.number):
        return "card number failed Luhn check (invalid card number)"
    if not card.expiry_month or not card.expiry_year:
        return "expiry date is missing"
    try:
        today = datetime.today()
        if card.expiry_year < today.year or (
            card.expiry_year == today.year and card.expiry_month < today.month
        ):
            return "card has expired"
    except Exception:
        return "expiry date is invalid"
    if not card.cvv:
        return "CVV is missing"
    if not re.match(r"^\d{3,4}$", card.cvv):
        return "CVV must be 3 or 4 digits"
    if not card.cardholder_name:
        return "cardholder name is missing"
    return None

def redact_user_input(text: str, state: Any) -> str:
    """
    Finds the clean values already stored in state and redacts them from the text,
    replacing them with [True] flags before the text is saved to history.
    """
    sanitized = text

    # 1. Redact Card Number
    if state.card.number:
        # Match the card number digits with optional spaces/dashes
        pattern = r"\b" + r"[ -]*?".join(state.card.number) + r"\b"
        sanitized = re.sub(pattern, "[CARD_NUMBER: True]", sanitized)
        sanitized = sanitized.replace(state.card.number, "[CARD_NUMBER: True]")

    # 2. Redact DOB
    if state.provided_dob:
        year, month, day = state.provided_dob.split("-")
        # Replace common date formats matching these digits
        dob_patterns = [
            state.provided_dob,
            f"{day}-{month}-{year}",
            f"{day}/{month}/{year}",
            f"{int(day)}th May {year}",
        ]
        for pat in dob_patterns:
            sanitized = sanitized.replace(pat, "[DOB: True]")
        
        # Regex fallbacks for date patterns
        sanitized = re.sub(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b", "[DOB: True]", sanitized)
        sanitized = re.sub(r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}\b", "[DOB: True]", sanitized, flags=re.IGNORECASE)

    # 3. Redact Pincode
    if state.provided_pincode:
        pattern = r"\b" + r"\s*?".join(state.provided_pincode) + r"\b"
        sanitized = re.sub(pattern, "[PINCODE: True]", sanitized)
        sanitized = sanitized.replace(state.provided_pincode, "[PINCODE: True]")

    # 4. Redact Aadhaar Last 4
    if state.provided_aadhaar:
        sanitized = re.sub(rf"\b{state.provided_aadhaar}\b", "[AADHAAR: True]", sanitized)

    # 5. Redact CVV
    if state.card.cvv:
        sanitized = re.sub(rf"\b{state.card.cvv}\b", "[CVV: True]", sanitized)

    return sanitized

# ── SLM Gateway Router (Append to the end of extractor.py) ────────────────────

_CLASSIFY_SYSTEM_PROMPT = """
You are a fast, low-latency intent router for a payment collection system.
Your single job is to classify if the user's latest message contains useful, structured information required for the current step, or if it is just random chitchat, greetings, questions, comments, or out-of-scope text.

Active Step: {stage}

Information defined as useful for each step:
- "INIT" or "AWAIT_ACCOUNT_ID": Contains an Account ID (e.g., starts with 'ACC' followed by digits).
- "AWAIT_NAME" or "AWAIT_SECONDARY": Contains a full name, date of birth, pincode, or Aadhaar last 4.
- "AWAIT_AMOUNT": Contains a payment amount, numbers to pay, or intent to pay (e.g. "clear balance").
- "AWAIT_CARD": Contains card details (number, expiry, CVV, or cardholder name).

Examples of Chitchat/Out-of-Scope (has_useful_info = false):
- Greetings ("hello", "hi how are you"), casual remarks ("no hi?", "First tell me hello"), questions ("where is my CVV?"), or random text.

Return ONLY a single JSON object with the key "has_useful_info" (boolean):
{{"has_useful_info": true/false}}
"""


def classify_intent(text: str, stage: str) -> bool:
    """Classifies if the user input contains actionable step information."""
    try:
        llm = get_llm()
        messages = [
            SystemMessage(content=_CLASSIFY_SYSTEM_PROMPT.format(stage=stage)),
            HumanMessage(content=text),
        ]
        response = llm.invoke(messages)
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
        result = json.loads(raw)
        return bool(result.get("has_useful_info", False))
    except Exception as exc:
        log.warning("Intent routing failed, defaulting to True: %s", exc)
        return True