#!/usr/bin/env python3
"""
Proper Multi-turn Rerollout with Tool Call ID Handling

When model makes a tool call:
1. Model generates NEW tool_call_id
2. We update the tool response to use this NEW id
3. Context stays coherent

Usage:
    uv run python scripts/rerollout_proper.py parsed_datasets/interactive_agent_parsed.jsonl -n 1 -v
"""

import argparse
import json

import requests


def rerollout_record(record: dict, api_url: str, model: str, verbose: bool = False) -> dict:
    """
    Proper rerollout that handles tool_call_id mapping.
    """
    original_messages = record.get("messages", [])
    tools = record.get("tools", [])

    context = []  # What we send to model
    new_messages = []  # Final output

    # Track tool_call_id mapping: original_id -> new_id
    tool_id_map = {}

    i = 0
    while i < len(original_messages):
        msg = original_messages[i]
        role = msg.get("role")

        if role == "system" or role == "user":
            # Keep as-is
            clean_msg = {"role": role, "content": msg.get("content", "")}
            context.append(clean_msg)
            new_messages.append(clean_msg)
            if verbose:
                print(f"[{role}] Added to context")
            i += 1

        elif role == "assistant":
            if verbose:
                print("[assistant] Regenerating with model...")
                print(f"  Context size: {len(context)} messages")

            # Call model
            payload = {
                "model": model,
                "messages": context,
                "tools": tools if tools else None,
                "temperature": 0.7,
                "max_tokens": 2048,
            }
            payload = {k: v for k, v in payload.items() if v is not None}

            try:
                resp = requests.post(api_url, json=payload, timeout=120)
                resp.raise_for_status()
                result = resp.json()
                new_assistant = result["choices"][0]["message"]
            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}")
                # Fallback to original
                new_assistant = msg

            # Check if model made tool calls
            new_tool_calls = new_assistant.get("tool_calls", [])
            orig_tool_calls = msg.get("tool_calls", [])

            if verbose:
                if new_tool_calls:
                    print(f"  -> Tool call: {new_tool_calls[0]['function']['name']}")
                    print(f"     New ID: {new_tool_calls[0].get('id', 'N/A')}")
                else:
                    content = new_assistant.get("content", "")[:80]
                    print(f"  -> Content: {content}...")

            # Build clean assistant message
            clean_assistant = {
                "role": "assistant",
                "content": new_assistant.get("content") or ""
            }
            if new_tool_calls:
                clean_assistant["tool_calls"] = new_tool_calls

                # Map original tool_call_ids to new ones
                for j, new_tc in enumerate(new_tool_calls):
                    if j < len(orig_tool_calls):
                        orig_id = orig_tool_calls[j].get("id")
                        new_id = new_tc.get("id")
                        if orig_id and new_id:
                            tool_id_map[orig_id] = new_id
                            if verbose:
                                print(f"  Mapped tool_call_id: {orig_id[:20]}... -> {new_id[:20]}...")

            context.append(clean_assistant)
            new_messages.append(clean_assistant)
            i += 1

            # If model made tool calls, we need to include tool responses
            # Look ahead for tool messages that correspond to original tool calls
            if new_tool_calls:
                while i < len(original_messages) and original_messages[i].get("role") == "tool":
                    tool_msg = original_messages[i]
                    orig_tool_id = tool_msg.get("tool_call_id")

                    # Map to new tool_call_id
                    new_tool_id = tool_id_map.get(orig_tool_id, new_tool_calls[0].get("id"))

                    clean_tool = {
                        "role": "tool",
                        "tool_call_id": new_tool_id,
                        "content": tool_msg.get("content", "")
                    }

                    if verbose:
                        print(f"[tool] Using mapped ID: {new_tool_id[:30] if new_tool_id else 'N/A'}...")

                    context.append(clean_tool)
                    new_messages.append(clean_tool)
                    i += 1
            else:
                # Model didn't make tool call but original did - skip original tool responses
                while i < len(original_messages) and original_messages[i].get("role") == "tool":
                    if verbose:
                        print("[tool] SKIPPED (model didn't make tool call)")
                    i += 1

        elif role == "tool":
            # Shouldn't reach here if logic above is correct, but handle gracefully
            if verbose:
                print("[tool] Unexpected tool message, skipping")
            i += 1

        else:
            i += 1

    return {
        "uuid": record.get("uuid"),
        "messages": new_messages,
        "tools": tools,
        "license": record.get("license"),
        "used_in": record.get("used_in", []),
    }


def main():
    parser = argparse.ArgumentParser(description="Proper multi-turn rerollout with tool_call_id handling")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("-o", "--output", help="Output JSONL file")
    parser.add_argument("-n", "--num", type=int, default=1, help="Number of records")
    parser.add_argument("-i", "--index", type=int, help="Specific record index")
    parser.add_argument("--api-url", default="http://localhost:30000/v1/chat/completions")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3.2")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--proof", help="Write before/after proof file")

    args = parser.parse_args()

    # Read input
    records = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records")

    # Select records
    if args.index is not None:
        to_process = [(args.index, records[args.index])]
    else:
        to_process = list(enumerate(records[:args.num]))

    print(f"Processing {len(to_process)} record(s)")
    print(f"Model: {args.model}")
    print()

    results = []
    for idx, original in to_process:
        print(f"=== Record {idx}: {original.get('uuid', 'unknown')[:12]}... ===")

        rerolled = rerollout_record(
            original,
            api_url=args.api_url,
            model=args.model,
            verbose=args.verbose
        )
        results.append((original, rerolled))
        print()

    # Write proof file if requested
    if args.proof:
        original, rerolled = results[0]
        proof = {
            "description": "Rerollout Proof with Proper Tool Call ID Handling",
            "model": args.model,
            "uuid": original.get("uuid"),
            "BEFORE": original,
            "AFTER": rerolled
        }
        with open(args.proof, "w") as f:
            json.dump(proof, f, indent=2)
        print(f"Proof written to {args.proof}")

    # Write output
    if args.output:
        with open(args.output, "w") as f:
            for _, rerolled in results:
                f.write(json.dumps(rerolled) + "\n")
        print(f"Output written to {args.output}")
    else:
        # Print summary
        for original, rerolled in results:
            print("=== COMPARISON ===")
            print("\nBEFORE (assistant messages):")
            for m in original["messages"]:
                if m["role"] == "assistant":
                    tc = m.get("tool_calls", [])
                    print(f"  content: \"{m.get('content', '')[:60]}\"")
                    if tc:
                        print(f"  tool_call: {tc[0]['function']['name']} (id: {tc[0].get('id', 'N/A')[:20]}...)")

            print("\nAFTER (assistant messages):")
            for m in rerolled["messages"]:
                if m["role"] == "assistant":
                    tc = m.get("tool_calls", [])
                    print(f"  content: \"{m.get('content', '')[:60]}\"")
                    if tc:
                        print(f"  tool_call: {tc[0]['function']['name']} (id: {tc[0].get('id', 'N/A')[:20]}...)")


if __name__ == "__main__":
    main()
