"""
Adversarial test cases designed to break the Payment Collection Agent.
Run with: python eval/eval_forced_break.py
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List

# Import your existing evaluator
sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.test_cases import TestCase
from eval.evaluator import run_test_case

ADVERSARIAL_CASES: List[TestCase] = [

    # ── Original 3 Test Cases ─────────────────────────────────────────────────

    TestCase(
        name="adversarial_amount_change",
        description="User tries to change the payment amount mid-flow. Agent should ideally adapt.",
        turns=[
            "ACC1001",
            "Nithin Jain",
            "1990-05-14",
            "I want to pay 500",
            "Wait, actually change the amount to 200",
            "4532015112830366",
            "12/2027",
            "123",
        ],
        expected_final_stage="DONE",
        must_contain_any=["200"],
        tags=["adversarial", "immutability"],
    ),

    TestCase(
        name="adversarial_account_typo_correction",
        description="User corrects their account ID in the same or next sentence.",
        turns=[
            "My account is ACC9999. Oh wait, sorry, it's ACC1001.",
            "Nithin Jain",
            "1990-05-14",
        ],
        expected_final_stage="AWAIT_AMOUNT",
        tags=["adversarial", "immutability"],
    ),

    TestCase(
        name="adversarial_number_confusion",
        description="Providing an amount that matches Aadhaar length, and a CVV early.",
        turns=[
            "ACC1004",
            "Rahul Mehta",
            "My aadhaar ends in 1357 and I want to pay 1357 rupees.",
            "Here is my card 4532015112830366 expires 12/2027 cvv 123",
        ],
        expected_final_stage="DONE",
        tags=["adversarial", "extraction_confusion"],
    ),

    # ── 10 New Advanced Adversarial Cases ─────────────────────────────────────

    TestCase(
        name="adversarial_cvv_leading_zeros",
        description="User's card CVV contains leading zeros (e.g. 007). Tests integer-casting bugs.",
        turns=[
            "ACC1001",
            "Nithin Jain",
            "1990-05-14",
            "500",
            "4532015112830366",
            "12/2027",
            "cvv is 007",  # CVV with leading zeros!
        ],
        expected_final_stage="DONE",
        tags=["adversarial", "cvv_truncation"],
    ),

    TestCase(
        name="adversarial_name_punctuation_casing",
        description="User enters their name with trailing punctuation. Tests exact case regex bounds.",
        turns=[
            "ACC1001",
            "My name is Nithin Jain.",  # Trailing period!
            "1990-05-14",
            "500",
            "4532015112830366 exp 12/2027 cvv 123",
        ],
        expected_final_stage="DONE",
        tags=["adversarial", "name_punctuation"],
    ),

    TestCase(
        name="adversarial_insistent_zero_balance",
        description="User insists on paying an account that already has a zero balance.",
        turns=[
            "ACC1003",  # Priya Agarwal (balance ₹0.00)
            "Priya Agarwal",
            "I still want to pay 500 rupees",  # Insistent turn after session is CLOSED
        ],
        expected_final_stage="DONE",  # Must maintain Stage.DONE
        must_contain_any=["zero", "no balance", "nothing to pay"],
        tags=["adversarial", "zero_balance_abuse"],
    ),

    TestCase(
        name="adversarial_double_pincode_correction",
        description="User self-corrects their pincode in a single conversational sentence.",
        turns=[
            "ACC1001",
            "Nithin Jain",
            "My old pincode was 400002, but my current pincode is 400001.",  # Self-correction
            "500",
            "4532015112830366 exp 12/2027 cvv 123",
        ],
        expected_final_stage="DONE",
        tags=["adversarial", "double_pincode"],
    ),

    TestCase(
        name="adversarial_micro_payment_attack",
        description="User enters a fractional payment (0.004) which rounds down to zero.",
        turns=[
            "ACC1001",
            "Nithin Jain",
            "1990-05-14",
            "I want to pay 0.004 rupees",  # Rounds down to ₹0.00 (invalid)
        ],
        expected_final_stage="AWAIT_AMOUNT",  # Prompts again for a valid amount
        tags=["adversarial", "micro_payment"],
    ),

    TestCase(
        name="adversarial_cardholder_name_default",
        description="User omits cardholder name to test secure defaulting logic.",
        turns=[
            "ACC1001",
            "Nithin Jain",
            "1990-05-14",
            "500",
            "card number 4532015112830366 expiry 12/2027 cvv 123",  # No cardholder name!
        ],
        expected_final_stage="DONE",
        tags=["adversarial", "cardholder_default"],
    ),

    TestCase(
        name="adversarial_invalid_leap_year_dob",
        description="User enters a non-existent leap day birthdate (1989-02-29).",
        turns=[
            "ACC1004",  # Rahul Mehta (DOB 1988-02-29)
            "Rahul Mehta",
            "My DOB is 1989-02-29",  # Invalid calendar day (1989 is not leap year)
        ],
        expected_final_stage="AWAIT_SECONDARY",  # Rejects and re-prompts
        tags=["adversarial", "invalid_leap_year"],
    ),

    TestCase(
        name="adversarial_mid_flow_account_swapping",
        description="User tries to change account ID after starting verification. System must adapt.",
        turns=[
            "ACC1001",
            "Nithin Jain",
            "Wait, change my account to ACC1002",  # Attempt to swap unverified account
            "Rajarajeswari Balasubramaniam",       # Must verify ACC1002 now!
            "DOB is 1985-11-23",
        ],
        expected_final_stage="AWAIT_AMOUNT",  # Correctly swaps and verifies ACC1002!
        tags=["adversarial", "account_swapping"],
    ),

    TestCase(
        name="adversarial_cvv_like_payment_amount",
        description="User pays exactly ₹123.00, testing amount-CVV extraction boundaries.",
        turns=[
            "ACC1001",
            "Nithin Jain",
            "1990-05-14",
            "123",  # Amount is 123 (matching CVV length)
        ],
        expected_final_stage="AWAIT_CARD",  # Successfully sets amount to 123, proceeds to ask for card
        tags=["adversarial", "amount_cvv_collision"],
    ),

    TestCase(
        name="adversarial_unrecoverable_api_failure",
        description="User enters an amount that triggers an unrecoverable business error.",
        turns=[
            "ACC1002",
            "Rajarajeswari Balasubramaniam",
            "1985-11-23",
            "9999",  # Exceeds the account balance of 540.00
        ],
        expected_final_stage="AWAIT_AMOUNT",  # Gracefully blocks and asks to re-enter
        tags=["adversarial", "exceed_balance"],
    ),
]

if __name__ == "__main__":
    print(f"Running {len(ADVERSARIAL_CASES)} adversarial test cases...\n")
    failed = 0
    
    for tc in ADVERSARIAL_CASES:
        print(f"Testing: {tc.name}...")
        result = run_test_case(tc, verbose=False)
        
        # We define a "break" as the code failing to reach our ideal expected stage
        # OR missing expected text content (like the changed amount)
        if not result.passed:
            print(f"  [SUCCESS] You broke the agent! It failed the test case.")
            if not result.stage_ok:
                print(f"    -> Expected Stage: {tc.expected_final_stage}, Got: {result.final_stage}")
            if result.content_ok is False:
                print(f"    -> Agent's text did not contain expected terms: {tc.must_contain_any}")
            print("-" * 60)
        else:
            print(f"  [FAILED TO BREAK] The agent survived this adversarial attack.")
            failed += 1
            print("-" * 60)
            
    print(f"\nAdversarial Summary: {len(ADVERSARIAL_CASES) - failed} vulnerabilities exposed, {failed} defended.")