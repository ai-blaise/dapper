#!/usr/bin/env python3
"""
Multi-turn Rerollout Script

Regenerates assistant responses using a new model while maintaining
conversation context. For each assistant turn:
1. Send all prior context to the model
2. Get model's new response
3. Use that response (not original) as context for next turns

Usage:
    python scripts/rerollout.py parsed_datasets/interactive_agent_parsed.jsonl -n 1
    python scripts/rerollout.py parsed_datasets/interactive_agent_parsed.jsonl -i 0 --verbose
"""

import argparse
import json
from pathlib import Path

import requests


def rerollout_record(record: dict, api_url: str, model: str, verbose: bool = False) -> dict:
    """
    Rerollout a single record, regenerating all assistant responses.

    Args:
        record: Parsed record with messages, tools, etc.
        api_url: OpenAI-compatible API endpoint
        model: Model name/path
        verbose: Print detailed progress

    Returns:
        New record with regenerated assistant responses
    """
    messages = record.get("messages", [])
    tools = record.get("tools", [])

    # Build new conversation with regenerated assistant turns
    context = []  # Messages sent to model (accumulates)
    new_messages = []  # Final output messages
    turn = 0

    for i, msg in enumerate(messages):
        role = msg.get("role")

        if role == "assistant":
            turn += 1
            if verbose:
                print(f"  Turn {turn}: Regenerating assistant response...")
                print(f"    Context: {len(context)} messages")

            # Call model with current context
            payload = {
                "model": model,
                "messages": context,
                "tools": tools if tools else None,
                "temperature": 0.7,
                "max_tokens": 4096,
            }
            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}

            try:
                resp = requests.post(api_url, json=payload, timeout=120)
                resp.raise_for_status()
                result = resp.json()

                new_assistant = result["choices"][0]["message"]

                if verbose:
                    if new_assistant.get("tool_calls"):
                        tc = new_assistant["tool_calls"][0]
                        print(f"    -> Tool call: {tc['function']['name']}")
                    else:
                        content = new_assistant.get("content", "")[:100]
                        print(f"    -> Content: {content}...")

                # Add to context and output
                context.append(new_assistant)
                new_messages.append(new_assistant)

            except Exception as e:
                if verbose:
                    print(f"    ERROR: {e}")
                # On error, keep original (fallback)
                context.append(msg)
                new_messages.append(msg)

        else:
            # system, user, tool - keep as-is
            # Clean message for context
            clean_msg = {"role": role, "content": msg.get("content", "")}
            if msg.get("tool_call_id"):
                clean_msg["tool_call_id"] = msg["tool_call_id"]
            if msg.get("tool_calls"):
                clean_msg["tool_calls"] = msg["tool_calls"]
            if msg.get("name"):
                clean_msg["name"] = msg["name"]

            context.append(clean_msg)
            new_messages.append(clean_msg)

            if verbose:
                print(f"  [{role}] Added to context")

    # Build new record
    new_record = {
        "uuid": record.get("uuid"),
        "messages": new_messages,
        "tools": tools,
        "license": record.get("license"),
        "used_in": record.get("used_in", []),
    }
    if record.get("reasoning"):
        new_record["reasoning"] = record["reasoning"]

    return new_record


def main():
    parser = argparse.ArgumentParser(description="Multi-turn rerollout")
    parser.add_argument("input", help="Input JSONL file (parsed dataset)")
    parser.add_argument("-o", "--output", help="Output JSONL file")
    parser.add_argument("-n", "--num", type=int, default=1, help="Number of records to process")
    parser.add_argument("-i", "--index", type=int, help="Process specific record index")
    parser.add_argument("--api-url", default="http://localhost:30000/v1/chat/completions",
                        help="API endpoint")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3.2",
                        help="Model name")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Read input
    input_path = Path(args.input)
    records = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records from {input_path}")

    # Select records to process
    if args.index is not None:
        to_process = [(args.index, records[args.index])]
    else:
        to_process = [(i, r) for i, r in enumerate(records[:args.num])]

    print(f"Processing {len(to_process)} record(s)")
    print(f"API: {args.api_url}")
    print(f"Model: {args.model}")
    print()

    # Process
    results = []
    for idx, record in to_process:
        print(f"=== Record {idx}: {record.get('uuid', 'unknown')[:8]}... ===")
        msg_count = len(record.get("messages", []))
        assistant_count = sum(1 for m in record.get("messages", []) if m.get("role") == "assistant")
        print(f"  Messages: {msg_count}, Assistant turns: {assistant_count}")

        new_record = rerollout_record(
            record,
            api_url=args.api_url,
            model=args.model,
            verbose=args.verbose
        )
        results.append(new_record)
        print()

    # Output
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in results)
        print(f"Wrote {len(results)} records to {output_path}")
    else:
        # Print to stdout
        print("=== Results ===")
        for r in results:
            print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
