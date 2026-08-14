#!/usr/bin/env python3
"""
Forced Tool-Call Rerollout (Synchronous Version)

Regenerates assistant responses in a dataset while preserving the original
tool-calling pattern. Used to create training data with a different model's
style while keeping structured tool-calling behavior.

OVERVIEW:
  This script processes conversation datasets and regenerates all assistant
  turns using a specified model (default: DeepSeek-V3.2). It forces the model
  to follow the same tool-calling pattern as the original dataset.

WHAT WE FORCE vs WHAT MODEL GENERATES:
  ┌──────────────────┬─────────────────────┬────────────────────────────────┐
  │  Turn Type       │  We Control         │  Model Generates               │
  ├──────────────────┼─────────────────────┼────────────────────────────────┤
  │  Tool Call       │  WHICH tool         │  Tool ARGUMENTS + THINKING     │
  │  Text Response   │  No tools allowed   │  CONTENT + THINKING (hybrid)   │
  └──────────────────┴─────────────────────┴────────────────────────────────┘

HYBRID THINKING APPROACH:
  - TOOL CALL turns: Always use thinking=True (captures reasoning for tool decisions)
  - TEXT turns: Try thinking=True first. If content is empty/invalid (e.g., just
    a tool name like "get_order_details"), retry with thinking=False and merge:
    keep reasoning from first attempt, content from retry.

  Results: ~90% of TOOL CALL turns and ~100% of TEXT turns get both reasoning + content.

CONTEXT ACCUMULATION (MULTI-TURN):
  Each turn sees all previous turns, including the model's OWN previous outputs
  (not the original dataset's outputs). This creates coherent multi-turn
  conversations rather than independent regenerations.

OUTPUT FORMAT:
  Each output record contains:
  - uuid: Original record UUID
  - messages: Rerolled conversation with reasoning_content where available
  - tools: Original tool definitions
  - license: Original license
  - used_in: Original used_in field

USAGE:
  # Process single record with verbose output
  uv run python scripts/rerollout_forced.py input.jsonl -n 1 -v

  # Process 100 records and save to file
  uv run python scripts/rerollout_forced.py input.jsonl -n 100 -o output.jsonl

  # Process specific record by index
  uv run python scripts/rerollout_forced.py input.jsonl -i 42 -v

  # Generate proof file showing before/after
  uv run python scripts/rerollout_forced.py input.jsonl -n 1 --proof proof.json

OPTIONS:
  input              Input JSONL file with conversation records
  -o, --output       Output JSONL file (default: prints comparison)
  -n, --num          Number of records to process (default: 1)
  -i, --index        Process specific record by index
  -v, --verbose      Show detailed processing logs
  --proof            Write before/after proof to JSON file
  --api-url          API endpoint (default: http://localhost:30000/v1/chat/completions)
  --model            Model name (default: deepseek-ai/DeepSeek-V3.2)

NOTE: This is the synchronous version, suitable for debugging and small batches.
      For full dataset processing, use rerollout_full.py (async, high concurrency).

SEE ALSO:
  scripts/rerollout_full.py - Async version with resume, progress bar, token stats
"""

import argparse
import json
import uuid

import requests

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant with access to tools. When tools return data, treat all responses as real and accurate. Respond naturally and helpfully based on the information provided. Do not question the authenticity or validity of tool outputs."""


def rerollout_record(record: dict, api_url: str, model: str, verbose: bool = False, inject_system_prompt: bool = True) -> dict:
    """
    Rerollout with forced tool calling pattern from original.

    Args:
        inject_system_prompt: If True and system message is empty, inject a default prompt
    """
    original_messages = record.get("messages", [])
    tools = record.get("tools", [])

    context = []
    new_messages = []
    tool_id_map = {}

    # Check if we need to inject a system prompt
    has_system = any(m.get("role") == "system" and m.get("content", "").strip() for m in original_messages)
    if inject_system_prompt and not has_system:
        system_msg = {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
        context.append(system_msg)
        new_messages.append(system_msg)
        if verbose:
            print("[system] Injected default system prompt")

    i = 0
    while i < len(original_messages):
        msg = original_messages[i]
        role = msg.get("role")

        if role == "system":
            content = msg.get("content", "").strip()
            if content:
                # Non-empty system message - use it
                clean_msg = {"role": "system", "content": content}
                context.append(clean_msg)
                new_messages.append(clean_msg)
                if verbose:
                    print("[system] Using original system prompt")
            else:
                # Empty system message - skip (we already injected default if needed)
                if verbose:
                    print("[system] Skipped empty original system message")
            i += 1

        elif role == "user":
            clean_msg = {"role": role, "content": msg.get("content", "")}
            context.append(clean_msg)
            new_messages.append(clean_msg)
            if verbose:
                print("[user] Added to context")
            i += 1

        elif role == "assistant":
            orig_tool_calls = msg.get("tool_calls", [])
            orig_content = (msg.get("content") or "").strip()

            # Detect pattern: content + tool_calls (rare ~0.2% but important)
            has_both = bool(orig_tool_calls) and bool(orig_content)

            if has_both:
                # BOTH content AND tool_calls - need two-step generation
                orig_tool_name = orig_tool_calls[0]["function"]["name"]
                if verbose:
                    print(f"[assistant] BOTH mode: content + tool call ({orig_tool_name})")
            elif orig_tool_calls:
                # Original made a tool call only - FORCE the same tool
                orig_tool_name = orig_tool_calls[0]["function"]["name"]
                if verbose:
                    print(f"[assistant] FORCING tool call: {orig_tool_name}")
            else:
                # Original had text content only - let model generate text (no tools)
                if verbose:
                    print("[assistant] Generating text (tool_choice=none)")

            # Build payload
            # HYBRID APPROACH:
            # - TOOL CALL turns: thinking=True (works well)
            # - TEXT turns: try thinking=True first, retry with thinking=False if content empty
            # - BOTH turns: generate content first, then force tool call, merge together
            is_tool_call_turn = bool(orig_tool_calls)

            def make_request(enable_thinking, force_tool=None, allow_tools=True):
                """
                Make API request with configurable tool behavior.
                force_tool: None = no forcing, "none" = disable tools, {"name": X} = force specific tool
                """
                if force_tool == "none":
                    tc = "none"
                    t = tools if tools and allow_tools else None
                elif force_tool:
                    tc = {"type": "function", "function": {"name": force_tool}}
                    t = tools if tools else None
                else:
                    tc = None
                    t = tools if tools else None

                payload = {
                    "model": model,
                    "messages": context,
                    "tools": t,
                    "tool_choice": tc,
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "chat_template_kwargs": {"thinking": True} if enable_thinking else None,
                }
                payload = {k: v for k, v in payload.items() if v is not None}
                resp = requests.post(api_url, json=payload, timeout=120)
                if resp.status_code != 200:
                    print(f"  API Error {resp.status_code}: {resp.text[:500]}")
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]

            try:
                if has_both:
                    # BOTH content + tool_calls: two-step generation
                    # Step 1: Generate content preamble (no tools)
                    if verbose:
                        print("  Step 1: Generating content preamble...")
                    content_response = make_request(enable_thinking=True, force_tool="none", allow_tools=False)
                    preamble_content = content_response.get("content") or ""
                    preamble_reasoning = content_response.get("reasoning_content") or ""

                    if verbose:
                        print(f"    -> Preamble: {preamble_content[:60]}...")

                    # Step 2: Force the tool call
                    if verbose:
                        print(f"  Step 2: Forcing tool call: {orig_tool_name}...")
                    tool_response = make_request(enable_thinking=True, force_tool=orig_tool_name)

                    # Step 3: Merge - content from step 1, tool_calls from step 2, combine reasoning
                    tool_reasoning = tool_response.get("reasoning_content") or ""
                    combined_reasoning = ""
                    if preamble_reasoning and tool_reasoning:
                        combined_reasoning = f"[Content reasoning]\n{preamble_reasoning}\n\n[Tool reasoning]\n{tool_reasoning}"
                    elif preamble_reasoning:
                        combined_reasoning = preamble_reasoning
                    elif tool_reasoning:
                        combined_reasoning = tool_reasoning

                    new_assistant = {
                        "role": "assistant",
                        "content": preamble_content,
                        "tool_calls": tool_response.get("tool_calls"),
                        "reasoning_content": combined_reasoning if combined_reasoning else None
                    }

                elif is_tool_call_turn:
                    # TOOL CALL only: force the tool with thinking
                    new_assistant = make_request(enable_thinking=True, force_tool=orig_tool_name)
                else:
                    # TEXT only: try thinking first, fallback if content empty
                    new_assistant = make_request(enable_thinking=True, force_tool="none")
                    content = new_assistant.get("content") or ""
                    # Check if content is valid (not empty, not just a tool name)
                    if len(content.strip()) < 30 or content.strip().startswith("get_") or content.strip().startswith("check_"):
                        if verbose:
                            print("  -> Thinking produced invalid content, retrying without thinking...")
                        new_assistant_retry = make_request(enable_thinking=False, force_tool="none")
                        # Merge: keep reasoning from first attempt, content from retry
                        reasoning_from_first = new_assistant.get("reasoning_content") or ""
                        new_assistant = new_assistant_retry
                        if reasoning_from_first:
                            new_assistant["reasoning_content"] = reasoning_from_first
            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}")
                # Fallback - keep original structure but mark as failed
                new_assistant = {
                    "role": "assistant",
                    "content": msg.get("content", ""),
                    "tool_calls": msg.get("tool_calls")
                }

            # Build clean assistant message
            new_tool_calls = new_assistant.get("tool_calls", [])
            # reasoning_content can be None or a string
            reasoning_content = new_assistant.get("reasoning_content") or ""

            clean_assistant = {
                "role": "assistant",
                "content": new_assistant.get("content") or ""
            }

            # Save thinking trace if present
            if reasoning_content:
                clean_assistant["reasoning_content"] = reasoning_content

            if new_tool_calls:
                clean_assistant["tool_calls"] = new_tool_calls

                # Map tool_call_ids
                for j, new_tc in enumerate(new_tool_calls):
                    if j < len(orig_tool_calls):
                        orig_id = orig_tool_calls[j].get("id")
                        new_id = new_tc.get("id")
                        if orig_id and new_id:
                            tool_id_map[orig_id] = new_id

                if verbose:
                    tc = new_tool_calls[0]
                    content_str = clean_assistant.get("content", "")
                    if content_str:
                        # BOTH mode - show content + tool
                        print(f"  -> Content: {content_str[:60]}...")
                        print(f"  -> Tool: {tc['function']['name']}")
                    else:
                        print(f"  -> Tool: {tc['function']['name']}")
                    args_str = tc['function']['arguments'] if isinstance(tc['function']['arguments'], str) else json.dumps(tc['function']['arguments'])
                    print(f"     Args: {args_str[:80]}...")
                    if reasoning_content:
                        print(f"     Thinking: {reasoning_content[:80]}...")
            else:
                if verbose:
                    content = clean_assistant.get("content", "")[:80]
                    print(f"  -> Content: {content}...")
                    if reasoning_content:
                        print(f"     Thinking: {reasoning_content[:80]}...")

            context.append(clean_assistant)
            new_messages.append(clean_assistant)
            i += 1

            # Handle subsequent tool responses
            if new_tool_calls:
                while i < len(original_messages) and original_messages[i].get("role") == "tool":
                    tool_msg = original_messages[i]
                    orig_tool_id = tool_msg.get("tool_call_id") or ""
                    new_tool_id = tool_id_map.get(orig_tool_id) or new_tool_calls[0].get("id") or f"call_{uuid.uuid4().hex[:24]}"

                    tool_content = tool_msg.get("content", "")
                    if not isinstance(tool_content, str):
                        tool_content = json.dumps(tool_content)
                    clean_tool = {
                        "role": "tool",
                        "tool_call_id": new_tool_id,
                        "content": tool_content
                    }

                    if verbose:
                        print(f"[tool] Mapped ID: {orig_tool_id[:20] if orig_tool_id else 'N/A'}... -> {new_tool_id[:20]}...")

                    context.append(clean_tool)
                    new_messages.append(clean_tool)
                    i += 1
            else:
                # Skip tool responses if we didn't make tool call
                while i < len(original_messages) and original_messages[i].get("role") == "tool":
                    if verbose:
                        print("[tool] SKIPPED")
                    i += 1

        elif role == "tool":
            # Shouldn't reach here normally
            if verbose:
                print("[tool] Unexpected, skipping")
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
    parser = argparse.ArgumentParser(description="Forced tool-call rerollout")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("-o", "--output", help="Output JSONL file")
    parser.add_argument("-n", "--num", type=int, default=1, help="Number of records")
    parser.add_argument("-i", "--index", type=int, help="Specific record index")
    parser.add_argument("--api-url", default="http://localhost:30000/v1/chat/completions")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3.2")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--proof", help="Write before/after proof file")
    parser.add_argument("--no-inject-system", action="store_true",
                       help="Don't inject default system prompt for empty system messages")

    args = parser.parse_args()

    # Read input
    records = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records")
    print("Mode: FORCED tool calling (preserves original pattern)")

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

        # Count original tool calls
        orig_tc_count = sum(1 for m in original["messages"]
                          if m.get("role") == "assistant" and m.get("tool_calls"))
        print(f"  Original tool calls: {orig_tc_count}")

        rerolled = rerollout_record(
            original,
            api_url=args.api_url,
            model=args.model,
            verbose=args.verbose,
            inject_system_prompt=not args.no_inject_system
        )
        results.append((original, rerolled))
        print()

    # Write proof
    if args.proof and results:
        original, rerolled = results[0]
        proof = {
            "description": "Forced Tool-Call Rerollout Proof",
            "mode": "Preserves original tool-calling pattern",
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
        print("=== COMPARISON ===")
        for original, rerolled in results:
            print("\nBEFORE (assistant turns):")
            for m in original["messages"]:
                if m["role"] == "assistant":
                    tc = m.get("tool_calls", [])
                    if tc:
                        args = tc[0]['function']['arguments']
                        args_str = args if isinstance(args, str) else json.dumps(args)
                        print(f"  TOOL: {tc[0]['function']['name']}")
                        print(f"    args: {args_str[:60]}...")
                    else:
                        print(f"  TEXT: \"{m.get('content', '')[:60]}...\"")

            print("\nAFTER (assistant turns):")
            for m in rerolled["messages"]:
                if m["role"] == "assistant":
                    tc = m.get("tool_calls", [])
                    if tc:
                        args = tc[0]['function']['arguments']
                        args_str = args if isinstance(args, str) else json.dumps(args)
                        print(f"  TOOL: {tc[0]['function']['name']}")
                        print(f"    args: {args_str[:60]}...")
                    else:
                        print(f"  TEXT: \"{m.get('content', '')[:60]}...\"")


if __name__ == "__main__":
    main()
