# parse_tester.py
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import extractor

def run_tests():
    # Ensure the testing folder exists
    output_dir = Path("testing")
    output_dir.mkdir(exist_ok=True)
    
    log_file_path = output_dir / "parsing_test_results.txt"
    
    # ── Test Suite Definition ──
    # Format: (input_string, extraction_type, optional_balance)
    # Types: account_id, name, dob, aadhaar, pincode, amount, card, secondary
    
    # 1. Official PDF Examples (Pages 1-5)
    pdf_tests = [
        # Account IDs
        ("ACC1001", "account_id", None),
        ("yeah my account number is ACC1001 I think", "account_id", None),
        ("it's ACC 1001", "account_id", None),
        ("account id: acc1001", "account_id", None),
        
        # Names
        ("Nithin Jain", "name", None),
        ("my name is Nithin Jain", "name", None),
        ("it's Nithin, Nithin Jain", "name", None),
        ("you can call me Raja but my full name is Rajarajeswari Balasubramaniam", "name", None),
        
        # DOBs
        ("1990-05-14", "dob", None),
        ("I was born on 14th May 1990", "dob", None),
        ("DOB is May 14, 90", "dob", None),
        ("14-05-1990", "dob", None),
        
        # Aadhaar / Pincode
        ("4321", "secondary", None),
        ("last four of my Aadhaar is 4321", "secondary", None),
        ("pincode? it's 4 0 0 0 0 1", "secondary", None),
        ("400001", "secondary", None),
        ("Aadhaar ends with 9876, shall I give pincode instead?", "secondary", None),
        
        # Amounts
        ("1000.00", "amount", 1250.75),
        ("I want to pay a thousand rupees", "amount", 1250.75),
        ("just clear the full amount", "amount", 1250.75),
        ("can I do 500 for now?", "amount", 1250.75),
        
        # Card Details
        ("the card number is 4532 0151 1283 0366", "card", None),
        ("expires December 2027", "card", None),
        ("12/27", "card", None),
        ("CVV is one two three", "card", None),
    ]
    
    # 2. 50 Structural Edge Cases (Designed to probe gaps and overlaps)
    edge_cases = [
        # Account ID Edge Cases (1-7)
        ("my account is ACC-1002, please check", "account_id", None),
        ("acc 1004", "account_id", None),
        ("the ID is ACC     1003", "account_id", None),
        ("My name is ACC, wait no, my ID is acc-1001", "account_id", None),
        ("Is the number acc1002?", "account_id", None),
        ("ACC1001", "account_id", None),
        ("please lookup acc  -  1004", "account_id", None),
        
        # Name Edge Cases (8-14)
        ("it's Rahul, Rahul Mehta", "name", None),
        ("you can call me Priya, full name Priya Agarwal", "name", None),
        ("name is Nithin Jain", "name", None),
        ("I am Rahul Mehta", "name", None),
        ("Nithin Jain is my name", "name", None),
        ("Rajarajeswari Balasubramaniam", "name", None),
        ("Priya Agarwal", "name", None),
        
        # DOB Edge Cases (15-22)
        ("1988-02-29", "dob", None), # Leap year date
        ("born on February 29, 1988", "dob", None),
        ("DOB 29/02/1988", "dob", None),
        ("my DOB is 23/11/1985", "dob", None),
        ("10th August 1992", "dob", None),
        ("August 10, 1992", "dob", None),
        ("born 1990-05-14", "dob", None),
        ("14th May 1990", "dob", None),
        
        # Aadhaar Edge Cases (23-28)
        ("4321", "secondary", None),
        ("Aadhaar ending in 9876", "secondary", None),
        ("my Aadhaar ends in 2468", "secondary", None),
        ("last 4 digits are 1357", "secondary", None),
        ("1357 is my Aadhaar last 4", "secondary", None),
        ("Aadhaar 2468", "secondary", None),
        
        # Pincode Edge Cases (29-34)
        ("400001", "secondary", None),
        ("pincode 4 0 0 0 0 2", "secondary", None),
        ("pincode: 400003", "secondary", None),
        ("it's 4 0 0 0 0 4", "secondary", None),
        ("my code is 400 001", "secondary", None),
        ("4 0 0 0 0 2", "secondary", None),
        
        # Payment Amount Edge Cases (35-40)
        ("five hundred", "amount", 1250.75),
        ("a thousand rupees", "amount", 1250.75),
        ("clear the full balance", "amount", 1250.75),
        ("clear everything please", "amount", 1250.75),
        ("pay 500", "amount", 1250.75),
        ("can I pay 1250.75?", "amount", 1250.75),
        
        # Card details Edge Cases (41-50)
        ("card 4532 0151 1283 0366", "card", None),
        ("4532-0151-1283-0366", "card", None),
        ("expiry 12/2027", "card", None),
        ("expires 12/27", "card", None),
        ("expiry is December 2027", "card", None),
        ("expiry 11/2028", "card", None),
        ("cvv is 123", "card", None),
        ("cvv is one two three", "card", None),
        ("cvv 321", "card", None),
        ("cvv 3 2 1", "card", None),
    ]

    all_tests = [("PDF Examples", pdf_tests), ("50 Custom Edge Cases", edge_cases)]
    
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("  EXTRACTOR PIPELINE DETAILED VERIFICATION LOGS\n")
        f.write("================================================================================\n")
        
        for section_title, tests in all_tests:
            f.write(f"\n=== SECTION: {section_title} ===\n\n")
            print(f"Running section: {section_title} ...")
            
            for idx, (text, test_type, balance) in enumerate(tests, 1):
                f.write(f"[{test_type.upper()}] Test {idx:02d} | Input: {repr(text)}\n")
                
                # Each extraction is executed statelessly (no dialogue history or existing card object)
                if test_type == "account_id":
                    res = extractor.extract_account_id(text)
                    f.write(f"      Parsed Result: {repr(res)}\n")
                elif test_type == "name":
                    res = extractor.extract_full_name(text)
                    f.write(f"      Parsed Result: {repr(res)}\n")
                elif test_type == "dob":
                    res = extractor.extract_dob(text)
                    f.write(f"      Parsed Result: {repr(res)}\n")
                elif test_type == "secondary":
                    res = extractor.extract_secondary_factor(text)
                    f.write(f"      Parsed Result (Dict): {repr(res)}\n")
                elif test_type == "amount":
                    res = extractor.extract_amount(text, balance=balance)
                    f.write(f"      Parsed Result: {repr(res)}  (Balance context: {balance})\n")
                elif test_type == "card":
                    res = extractor.extract_card_details(text)
                    f.write(f"      Parsed Result (CardDetails Object):\n")
                    f.write(f"         - Number: {repr(res.number)}\n")
                    f.write(f"         - Expiry Month: {repr(res.expiry_month)}\n")
                    f.write(f"         - Expiry Year: {repr(res.expiry_year)}\n")
                    f.write(f"         - CVV: {repr(res.cvv)}\n")
                    f.write(f"         - Cardholder Name: {repr(res.cardholder_name)}\n")
                
                f.write("--------------------------------------------------------------------------------\n")
                
    print(f"\nParsing tests completed. Log written to: {log_file_path}")

if __name__ == "__main__":
    run_tests()