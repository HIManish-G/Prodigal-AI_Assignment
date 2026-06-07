"""
Evaluation test cases.

Each TestCase defines a scripted user-turn sequence and the expected outcomes
that can be automatically verified.

Outcomes are checked by the EvaluationRunner:
  - expected_stage: final Stage the agent should reach
  - must_contain: substrings that MUST appear in any agent message
  - must_not_contain: strings that must NEVER appear (e.g. raw DOB, Aadhaar)
  - tool_calls_expected: list of tool names expected to have been called
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TestCase:
    name: str
    description: str
    turns: List[str]                     # user messages, one per turn
    expected_final_stage: str            # Stage enum value string
    must_contain_any: List[str] = field(default_factory=list)   # at least one hit
    must_not_expose: List[str] = field(default_factory=list)    # never in ANY agent msg
    check_txn_id: bool = False           # if True, last agent msg should have txn id
    tags: List[str] = field(default_factory=list)


TEST_CASES: List[TestCase] = [

    # ── Happy Path ────────────────────────────────────────────────────────────

    TestCase(
        name="happy_path_full_amount_dob",
        description="Clean end-to-end flow using DOB as secondary, paying full balance",
        turns=[
            "",                          # empty first message triggers greeting
            "ACC1001",
            "Nithin Jain",
            "DOB is 1990-05-14",
            "pay the full amount",
            "4532015112830366",
            "expires 12/2027",
            "cvv 123",
        ],
        expected_final_stage="DONE",
        check_txn_id=True,
        must_not_expose=["1990-05-14", "4321", "400001"],
        tags=["happy_path"],
    ),

    TestCase(
        name="happy_path_partial_aadhaar",
        description="Partial payment using Aadhaar last4 as secondary",
        turns=[
            "",
            "my account id is acc1001",
            "my name is Nithin Jain",
            "last four of my aadhaar is 4321",
            "I want to pay 500 rupees",
            "card number 4532 0151 1283 0366, expiry December 2027, cvv one two three",
        ],
        expected_final_stage="DONE",
        check_txn_id=True,
        must_not_expose=["1990-05-14", "4321", "400001"],
        tags=["happy_path", "partial_payment"],
    ),

    TestCase(
        name="happy_path_messy_inputs",
        description="Messy, colloquial user input throughout",
        turns=[
            "hey",
            "yeah my account number is ACC1001 I think",
            "it's Nithin, Nithin Jain",
            "pincode? it's 4 0 0 0 0 1",
            "just clear the full amount",
            "the card number is 4532 0151 1283 0366",
            "expires December 2027",
            "CVV is one two three",
        ],
        expected_final_stage="DONE",
        check_txn_id=True,
        tags=["happy_path", "messy_input"],
    ),

    TestCase(
        name="happy_path_all_in_one_card",
        description="User provides all card details in a single message",
        turns=[
            "",
            "ACC1002",
            "Rajarajeswari Balasubramaniam",
            "my dob is 23rd november 1985",
            "full amount please",
            "card 4532015112830366 expiry 11/2028 cvv 321 name Rajarajeswari Balasubramaniam",
        ],
        expected_final_stage="DONE",
        check_txn_id=True,
        tags=["happy_path", "long_name"],
    ),

    TestCase(
        name="happy_path_name_upfront",
        description="User volunteers name before being asked",
        turns=[
            "Hi, I'm Nithin Jain",
            "ACC1001",
            "1990-05-14",
            "500",
            "4532015112830366",
            "12/2027",
            "123",
        ],
        expected_final_stage="DONE",
        check_txn_id=True,
        tags=["happy_path", "out_of_order"],
    ),

    # ── Verification Failures ─────────────────────────────────────────────────

    TestCase(
        name="verification_wrong_name_lockout",
        description="User provides wrong name every attempt and gets locked out",
        turns=[
            "",
            "ACC1001",
            "John Smith",
            "1990-05-14",
            "Jane Doe",
            "4321",
            "Wrong Person",
            "400001",
        ],
        expected_final_stage="LOCKED_OUT",
        must_not_expose=["1990-05-14", "4321", "400001"],
        tags=["verification_failure", "lockout"],
    ),

    TestCase(
        name="verification_wrong_secondary_then_correct",
        description="Wrong secondary on first attempt, correct on second",
        turns=[
            "",
            "ACC1001",
            "Nithin Jain",
            "DOB is 1991-01-01",   # wrong DOB
            "Nithin Jain",
            "Aadhaar last 4 is 4321",  # correct Aadhaar
            "pay full amount",
            "4532015112830366 exp 12/2027 cvv 123",
        ],
        expected_final_stage="DONE",
        check_txn_id=True,
        tags=["verification_failure", "retry_success"],
    ),

    TestCase(
        name="verification_case_sensitive_name",
        description="Name with wrong casing should fail verification",
        turns=[
            "",
            "ACC1001",
            "nithin jain",           # lowercase — should fail
            "1990-05-14",
            "nithin jain",
            "4321",
            "nithin jain",
            "400001",
        ],
        expected_final_stage="LOCKED_OUT",
        tags=["verification_failure", "case_sensitive"],
    ),

    # ── Payment Failures ──────────────────────────────────────────────────────

    TestCase(
        name="payment_invalid_card_number",
        description="Card fails Luhn check — agent should reject before API call",
        turns=[
            "",
            "ACC1001",
            "Nithin Jain",
            "4321",
            "500",
            "1234567890123456",   # fails Luhn
            "4532015112830366",   # valid
            "12/2027",
            "123",
        ],
        expected_final_stage="DONE",
        check_txn_id=True,
        tags=["payment_failure", "invalid_card"],
    ),

    TestCase(
        name="payment_exceed_balance",
        description="User tries to pay more than the balance",
        turns=[
            "",
            "ACC1001",
            "Nithin Jain",
            "4321",
            "9999",               # exceeds 1250.75
            "500",
            "4532015112830366 exp 12/2027 cvv 123",
        ],
        expected_final_stage="DONE",
        check_txn_id=True,
        tags=["payment_failure", "excess_amount"],
    ),

    TestCase(
        name="payment_expired_card",
        description="User provides an expired card",
        turns=[
            "",
            "ACC1001",
            "Nithin Jain",
            "4321",
            "full amount",
            "4532015112830366",
            "01/2020",            # expired
            "123",
            "4532015112830366",
            "12/2027",
            "123",
        ],
        expected_final_stage="DONE",
        check_txn_id=True,
        tags=["payment_failure", "expired_card"],
    ),

    # ── Edge Cases ────────────────────────────────────────────────────────────

    TestCase(
        name="zero_balance_account",
        description="Account with zero balance — agent should close without payment",
        turns=[
            "",
            "ACC1003",
            "Priya Agarwal",
            "1992-08-10",
        ],
        expected_final_stage="DONE",
        must_contain_any=["zero", "no balance", "nothing to pay", "0"],
        tags=["edge_case", "zero_balance"],
    ),

    TestCase(
        name="leap_year_dob_acc1004",
        description="ACC1004 DOB is Feb 29 on a leap year — valid and should work",
        turns=[
            "",
            "ACC1004",
            "Rahul Mehta",
            "DOB is 1988-02-29",   # valid leap year date
            "full amount",
            "4532015112830366 exp 12/2027 cvv 123",
        ],
        expected_final_stage="DONE",
        check_txn_id=True,
        tags=["edge_case", "leap_year"],
    ),

    TestCase(
        name="invalid_account_id",
        description="User provides a non-existent account ID",
        turns=[
            "",
            "ACC9999",
            "ACC1001",
        ],
        expected_final_stage="AWAIT_NAME",   # recovered after valid ID
        tags=["edge_case", "invalid_account"],
    ),

    TestCase(
        name="spaced_account_id",
        description="User provides account ID with spaces or mixed case",
        turns=[
            "",
            "it's ACC 1001",
            "Nithin Jain",
            "4321",
            "500",
            "4532015112830366 exp 12/2027 cvv 123",
        ],
        expected_final_stage="DONE",
        check_txn_id=True,
        tags=["edge_case", "messy_input"],
    ),

    TestCase(
        name="long_name_raja",
        description="Long Indian name with nickname handling",
        turns=[
            "",
            "ACC1002",
            "you can call me Raja but my full name is Rajarajeswari Balasubramaniam",
            "DOB 23rd november 1985",
            "full amount",
            "4532015112830366 exp 11/2028 cvv 321",
        ],
        expected_final_stage="DONE",
        check_txn_id=True,
        tags=["edge_case", "long_name", "nickname"],
    ),
]


def get_test_case(name: str) -> Optional[TestCase]:
    for tc in TEST_CASES:
        if tc.name == name:
            return tc
    return None


def get_cases_by_tag(tag: str) -> List[TestCase]:
    return [tc for tc in TEST_CASES if tag in tc.tags]
