# Payment Collection AI Agent

A conversational AI agent that handles end-to-end payment collection — account lookup, identity verification, and card payment processing — over a natural language chat interface.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- A pulled Ollama model (default: `llama3.1:8b`)

---

## Setup

### 1. Install Ollama

Follow the instructions at https://ollama.com/download for your OS. Then pull the default model:

```bash
ollama pull llama3.1:8b
```

Ollama must be running before you start the agent. By default it listens on `http://localhost:11434`.

### 2. Clone the repository

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Configure environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model to use |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `PAYMENT_API_BASE_URL` | *(set in config.py)* | External payment API base URL |

---

## Running the Agent

### Interactive CLI

```bash
python cli.py
```

With debug logging:

```bash
python cli.py --debug
```

With a different model:

```bash
python cli.py --model qwen2.5:7b
```

### Programmatic usage

```python
from agent import Agent

agent = Agent()
print(agent.next("")["message"])          # greeting
print(agent.next("ACC1001")["message"])   # account lookup
print(agent.next("Nithin Jain")["message"])
```

---

## Recommended Models

Tested on a system with 8 GB VRAM:

| Model | VRAM | Notes |
|---|---|---|
| `llama3.1:8b` | ~4.5 GB | Default. Best tool-calling reliability. |
| `qwen2.5:7b` | ~4.5 GB | Strong instruction following. |
| `mistral-nemo` | ~7.5 GB | Higher quality, larger context. |

---

## Project Structure

```
agent.py          # Agent class — main interface (next() method)
extractor.py      # NLP extraction: account ID, name, DOB, card details, amounts
verifier.py       # Deterministic identity verification logic
responder.py      # LLM-powered natural language response generation
state.py          # Conversation state and stage definitions
tools.py          # External API wrappers (lookup-account, process-payment)
config.py         # Central configuration
cli.py            # Interactive CLI

eval/
  evaluator.py          # Automated test runner with deterministic checks + optional LLM judge
  test_cases.py         # Standard test cases (happy path, verification failure, payment failure, edge cases)
  eval_forced_break.py  # Adversarial test cases
```

---

## Running Evaluations

### Full test suite

```bash
python eval/evaluator.py
```

### Filter by tag

```bash
python eval/evaluator.py --tag happy_path
python eval/evaluator.py --tag verification_failure
python eval/evaluator.py --tag edge_case
```

### Specific test cases

```bash
python eval/evaluator.py --cases happy_path_full_amount_dob leap_year_dob_acc1004
```

### With LLM quality scoring

```bash
python eval/evaluator.py --llm-judge
```

### Save results to JSON

```bash
python eval/evaluator.py --output results.json
```

### Adversarial tests

```bash
python eval/eval_forced_break.py
```

---

## Test Accounts

| Account ID | Name | DOB | Aadhaar Last 4 | Pincode | Balance |
|---|---|---|---|---|---|
| ACC1001 | Nithin Jain | 1990-05-14 | 4321 | 400001 | ₹1,250.75 |
| ACC1002 | Rajarajeswari Balasubramaniam | 1985-11-23 | 9876 | 400002 | ₹540.00 |
| ACC1003 | Priya Agarwal | 1992-08-10 | 2468 | 400003 | ₹0.00 |
| ACC1004 | Rahul Mehta | 1988-02-29 | 1357 | 400004 | ₹3,200.50 |

> ACC1003 has a zero balance — the agent will close the session after verification without proceeding to payment.
> ACC1004 has a leap year DOB (1988-02-29) which is a valid calendar date and will be accepted.

---

## Verification Rules

Identity verification is implemented entirely in-agent (no separate API). A user is verified if:

- **Full name matches exactly** (case-sensitive, no fuzzy matching), AND
- **At least one secondary factor matches:**
  - Date of birth (YYYY-MM-DD)
  - Last 4 digits of Aadhaar
  - 6-digit pincode

Maximum **3 attempts** before the session is locked out.

---

## Architecture Overview

The agent is structured as a deterministic checklist state machine (`Stage` enum) with LLM used only for:

- **Extraction** (`extractor.py`) — parsing messy natural language into structured values
- **Response generation** (`responder.py`) — generating warm, natural replies

All verification logic (`verifier.py`), card validation, and amount validation are purely deterministic Python — no LLM in the critical path.

An **Inverted Gateway** pattern routes each turn: if the extraction sweep finds actionable data, the checklist runs. If not, the turn is classified as chitchat and routed to a conversational responder. This avoids unnecessary LLM calls on structured inputs and handles casual user messages gracefully.

---

## Assumptions

- `cardholder_name` on the payment request defaults to the account holder's name if the user does not provide one explicitly
- Partial payments (amount less than balance) are supported
- Account ID swaps are allowed before verification is complete; all state is cleared on swap
- Amounts with more than 2 decimal places are rounded, not rejected
