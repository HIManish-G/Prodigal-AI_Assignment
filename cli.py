#!/usr/bin/env python3
"""
Interactive CLI for the Payment Collection Agent.
Run:  python cli.py
      python cli.py --debug      (verbose logging)
      python cli.py --model qwen2.5:7b
"""
import argparse
import logging
import os
import sys

from agent import Agent


def run_interactive(debug: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("\n" + "=" * 60)
    print("  Payment Collection Agent  (type 'quit' or Ctrl-C to exit)")
    print("=" * 60 + "\n")

    agent = Agent()

    # Kick off the conversation
    first = agent.next("")
    print(f"Agent: {first['message']}\n")

    while True:
        try:
            user_in = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSession ended by user.")
            break

        if user_in.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        if not user_in:
            continue

        response = agent.next(user_in)
        print(f"\nAgent: {response['message']}\n")

        # Stop prompting once conversation is terminal
        if agent.state.stage.value in ("DONE", "LOCKED_OUT", "ERROR"):
            print("─" * 40)
            print("Conversation complete. Goodbye!")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Payment Collection Agent CLI")
    parser.add_argument("--debug",  action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--model", default=None,
        help="Override Ollama model (e.g. qwen2.5:7b). Default: llama3.1:8b"
    )
    args = parser.parse_args()

    if args.model:
        os.environ["OLLAMA_MODEL"] = args.model

    run_interactive(debug=args.debug)


if __name__ == "__main__":
    main()
