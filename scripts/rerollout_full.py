#!/usr/bin/env python3
"""
Forced Tool-Call Rerollout (Async Production Version)

High-performance async script for regenerating assistant responses across
entire datasets. Designed for production use with resume support, progress
tracking, and high concurrency.

OVERVIEW:
  This script processes conversation datasets and regenerates all assistant
  turns using a specified model (default: DeepSeek-V3.2). It forces the model
  to follow the same tool-calling pattern as the original dataset.

FEATURES:
  - Async processing with configurable concurrency (default: 3000 concurrent requests)
  - Automatic retry with exponential backoff for transient errors (timeouts, disconnects)
  - Resume support: safely resume interrupted runs without reprocessing
  - Progress bar with real-time token/s statistics
  - Incremental saving: results written immediately, safe against crashes
  - Verbose mode for debugging
  - Proof file generation for verification

HYBRID THINKING APPROACH:
  - TOOL CALL turns: Always use thinking=True (captures reasoning for tool decisions)
  - TEXT turns: Try thinking=True first. If content is empty/invalid (e.g., just
    a tool name like "get_order_details"), retry with thinking=False and merge:
    keep reasoning from first attempt, content from retry.

  Results: ~90% of TOOL CALL turns and ~100% of TEXT turns get both reasoning + content.

OUTPUT FORMAT:
  Each output record contains:
  - uuid: Original record UUID
  - messages: Rerolled conversation with reasoning_content where available
  - tools: Original tool definitions
  - license: Original license
  - used_in: Original used_in field

  Error records contain:
  - uuid: Record UUID
  - error: Error message
  - original: Original record for retry

USAGE:
  # Full dataset with default 3000 concurrent requests
  uv run python scripts/rerollout_full.py parsed_datasets/interactive_agent_parsed.jsonl

  # Custom concurrency (for rate limiting or resource constraints)
  uv run python scripts/rerollout_full.py input.jsonl -c 500

  # Resume interrupted run
  uv run python scripts/rerollout_full.py input.jsonl --resume

  # Limit records for testing
  uv run python scripts/rerollout_full.py input.jsonl -n 100

  # Process specific record with verbose output
  uv run python scripts/rerollout_full.py input.jsonl -i 42 -v

  # Generate proof file
  uv run python scripts/rerollout_full.py input.jsonl -n 1 --proof proof.json -v

OPTIONS:
  input              Input JSONL file with conversation records
  -o, --output       Output JSONL file (default: input_rerolled.jsonl)
  -n, --num          Limit number of records to process
  -i, --index        Process specific record by index only
  -c, --concurrency  Max concurrent requests (default: 3000)
  --resume           Resume from previous run (skip already processed UUIDs)
  -v, --verbose      Show detailed processing logs (disables progress bar)
  --proof            Write before/after proof to JSON file (first successful record)
  --api-url          API endpoint (default: http://localhost:30000/v1/chat/completions)
  --model            Model name (default: deepseek-ai/DeepSeek-V3.2)

PROGRESS OUTPUT:
  Rerolling: |████████████████████| 1000/1000 [05:23<00:00]
  ✓950 ✗50 | 1234 tok/s | total: 5.2M

  - ✓: Successful records
  - ✗: Failed records (saved with error info for retry)
  - tok/s: Completion tokens per second
  - total: Total tokens processed

COMPLETION SUMMARY:
  Shows records processed, success/error counts, timing, and token statistics.

SEE ALSO:
  scripts/rerollout_forced.py - Synchronous version for debugging and small batches
"""

import argparse
import asyncio
import json
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiofiles
import aiohttp
from tqdm.asyncio import tqdm

# Retry configuration (local GPU server - retry until success)
MAX_RETRIES = 0  # 0 = infinite retries
BASE_DELAY = 0.1  # seconds
MAX_DELAY = 2.0  # seconds

# Retryable exceptions
RETRYABLE_EXCEPTIONS = (
    asyncio.TimeoutError,
    aiohttp.ClientError,
    aiohttp.ServerDisconnectedError,
    ConnectionResetError,
    ConnectionError,
)


@dataclass
class TokenStats:
    """Thread-safe token statistics tracker."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0
    start_time: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, prompt: int, completion: int, total: int):
        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.total_tokens += total
            self.api_calls += 1

    def get_stats(self) -> dict:
        with self._lock:
            elapsed = time.time() - self.start_time
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "api_calls": self.api_calls,
                "elapsed": elapsed,
                "tokens_per_sec": self.total_tokens / elapsed if elapsed > 0 else 0,
                "completion_per_sec": self.completion_tokens / elapsed if elapsed > 0 else 0,
            }


async def rerollout_record(
    record: dict,
    api_url: str,
    model: str,
    token_stats: TokenStats,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    verbose: bool = False,
    max_retries: int = MAX_RETRIES
) -> dict:
    """
    Rerollout a single record with forced tool calling + thinking traces.
    """
    async with semaphore:
        original_messages = record.get("messages", [])
        tools = record.get("tools", [])

        context = []
        new_messages = []
        tool_id_map = {}

        i = 0
        while i < len(original_messages):
            msg = original_messages[i]
            role = msg.get("role")

            if role == "system" or role == "user":
                clean_msg = {"role": role, "content": msg.get("content", "")}
                context.append(clean_msg)
                new_messages.append(clean_msg)
                if verbose:
                    print(f"[{role}] Added to context")
                i += 1

            elif role == "assistant":
                orig_tool_calls = msg.get("tool_calls", [])

                # Determine tool_choice based on original behavior
                if orig_tool_calls:
                    orig_tool_name = orig_tool_calls[0]["function"]["name"]
                    tool_choice = {
                        "type": "function",
                        "function": {"name": orig_tool_name}
                    }
                    if verbose:
                        print(f"[assistant] FORCING tool call: {orig_tool_name}")
                else:
                    tool_choice = "none"
                    if verbose:
                        print("[assistant] Generating text (tool_choice=none)")

                # HYBRID APPROACH:
                # - TOOL CALL turns: thinking=True (works well)
                # - TEXT turns: try thinking=True first, retry with thinking=False if content empty
                is_tool_call_turn = bool(orig_tool_calls)

                async def make_request(
                    enable_thinking,
                    original_tool_calls=orig_tool_calls,
                    selected_tool_choice=tool_choice,
                ):
                    payload = {
                        "model": model,
                        "messages": context,
                        "tools": tools if tools and original_tool_calls else None,
                        "tool_choice": selected_tool_choice if tools else None,
                        "temperature": 0.7,
                        "max_tokens": 2048,
                        "chat_template_kwargs": {"thinking": True} if enable_thinking else None,
                    }
                    payload = {k: v for k, v in payload.items() if v is not None}

                    attempt = 0
                    while True:
                        try:
                            async with session.post(api_url, json=payload) as resp:
                                resp.raise_for_status()
                                result = await resp.json()
                            # Track token usage
                            usage = result.get("usage", {})
                            token_stats.add(
                                prompt=usage.get("prompt_tokens", 0),
                                completion=usage.get("completion_tokens", 0),
                                total=usage.get("total_tokens", 0)
                            )
                            return result["choices"][0]["message"]
                        except RETRYABLE_EXCEPTIONS as e:
                            attempt += 1
                            # max_retries=0 means infinite, otherwise give up after max_retries
                            if max_retries > 0 and attempt >= max_retries:
                                raise
                            # Exponential backoff with jitter, capped at MAX_DELAY
                            delay = min(BASE_DELAY * (2 ** min(attempt, 5)) + random.uniform(0, 0.5), MAX_DELAY)
                            if verbose:
                                print(f"  -> Retry {attempt} after {delay:.1f}s: {type(e).__name__}")
                            await asyncio.sleep(delay)

                if is_tool_call_turn:
                    # TOOL CALL: always use thinking
                    new_assistant = await make_request(enable_thinking=True)
                else:
                    # TEXT turn: try thinking first, fallback if content empty
                    new_assistant = await make_request(enable_thinking=True)
                    content = new_assistant.get("content") or ""
                    # Check if content is valid (not empty, not just a tool name)
                    if len(content.strip()) < 30 or content.strip().startswith("get_") or content.strip().startswith("check_"):
                        if verbose:
                            print("  -> Thinking produced invalid content, retrying without thinking...")
                        new_assistant_retry = await make_request(enable_thinking=False)
                        # Merge: keep reasoning from first attempt, content from retry
                        reasoning_from_first = new_assistant.get("reasoning_content") or ""
                        new_assistant = new_assistant_retry
                        if reasoning_from_first:
                            new_assistant["reasoning_content"] = reasoning_from_first

                # Build clean assistant message
                new_tool_calls = new_assistant.get("tool_calls", [])
                reasoning_content = new_assistant.get("reasoning_content") or ""

                clean_assistant = {
                    "role": "assistant",
                    "content": new_assistant.get("content") or ""
                }

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
                        print(f"  -> Tool: {tc['function']['name']}")
                        print(f"     Args: {tc['function']['arguments'][:80]}...")
                        if reasoning_content:
                            print(f"     Thinking: {reasoning_content[:80]}...")
                else:
                    if verbose:
                        content_preview = clean_assistant.get("content", "")[:80]
                        print(f"  -> Content: {content_preview}...")
                        if reasoning_content:
                            print(f"     Thinking: {reasoning_content[:80]}...")

                context.append(clean_assistant)
                new_messages.append(clean_assistant)
                i += 1

                # Handle subsequent tool responses
                if new_tool_calls:
                    while i < len(original_messages) and original_messages[i].get("role") == "tool":
                        tool_msg = original_messages[i]
                        orig_tool_id = tool_msg.get("tool_call_id")
                        new_tool_id = tool_id_map.get(orig_tool_id, new_tool_calls[0].get("id"))

                        clean_tool = {
                            "role": "tool",
                            "tool_call_id": new_tool_id,
                            "content": tool_msg.get("content", "")
                        }

                        if verbose:
                            print(f"[tool] Mapped ID: {orig_tool_id[:20]}... -> {new_tool_id[:20]}...")

                        context.append(clean_tool)
                        new_messages.append(clean_tool)
                        i += 1
                else:
                    while i < len(original_messages) and original_messages[i].get("role") == "tool":
                        if verbose:
                            print("[tool] SKIPPED")
                        i += 1

            elif role == "tool":
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


def load_and_clean_output(output_path: Path) -> tuple[set, set, list]:
    """Load processed records and separate successful from errors.

    Returns:
        (successful_uuids, error_uuids, successful_records)
        - successful_uuids: UUIDs to skip (already done)
        - error_uuids: UUIDs to retry
        - successful_records: list of successful record dicts to preserve
    """
    successful = set()
    errors = set()
    successful_records = []
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        uuid = record.get("uuid")
                        if uuid:
                            if "error" in record:
                                errors.add(uuid)
                            else:
                                successful.add(uuid)
                                successful_records.append(record)
                    except json.JSONDecodeError:
                        pass
    return successful, errors, successful_records


def format_tokens(n: int) -> str:
    """Format token count with K/M suffix."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


async def process_record(
    record: dict,
    api_url: str,
    model: str,
    token_stats: TokenStats,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    write_lock: asyncio.Lock,
    out_file,
    counters: dict,
    verbose: bool = False,
    max_retries: int = MAX_RETRIES
):
    """Process a single record and write result."""
    uuid = record.get("uuid", "unknown")

    try:
        rerolled = await rerollout_record(
            record, api_url, model, token_stats, session, semaphore, verbose, max_retries
        )

        async with write_lock:
            await out_file.write(json.dumps(rerolled) + "\n")
            await out_file.flush()

        counters["success"] += 1
        return True, uuid, rerolled

    except Exception as e:
        # Build descriptive error message (some exceptions like TimeoutError have empty str)
        error_msg = str(e) if str(e) else f"{type(e).__name__}: {e!r}"
        async with write_lock:
            error_record = {
                "uuid": uuid,
                "error": error_msg,
                "original": record
            }
            await out_file.write(json.dumps(error_record) + "\n")
            await out_file.flush()

        counters["error"] += 1
        return False, uuid, error_msg


async def main_async(args):
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_rerolled.jsonl"

    # Load input records
    print(f"Loading {input_path}...")
    records = []
    async with aiofiles.open(input_path) as input_file:
        async for line in input_file:
            if line.strip():
                records.append(json.loads(line))

    total_records = len(records)
    print(f"Loaded {total_records} records")

    # Handle specific index
    if args.index is not None:
        if args.index >= len(records):
            print(f"Error: Index {args.index} out of range (max {len(records)-1})")
            return
        records = [records[args.index]]
        print(f"Processing specific record at index {args.index}")
    # Limit if specified
    elif args.num:
        records = records[:args.num]
        print(f"Limited to {len(records)} records")

    # Check for resume
    successful_uuids = set()
    error_uuids = set()
    successful_records = []
    if args.resume and output_path.exists():
        successful_uuids, error_uuids, successful_records = load_and_clean_output(output_path)
        print(f"Resuming: {len(successful_uuids)} successful (skip), {len(error_uuids)} errors (retry)")

        # Rewrite output file with only successful records (removes old errors)
        if error_uuids:
            async with aiofiles.open(output_path, "w") as output_file:
                await output_file.writelines(
                    json.dumps(record) + "\n" for record in successful_records
                )
            print(f"Cleaned output file: removed {len(error_uuids)} error records for retry")

    # Filter out already successful records (errors will be retried)
    to_process = [r for r in records if r.get("uuid") not in successful_uuids]
    original_records = {r.get("uuid"): r for r in to_process}  # Keep originals for proof
    print(f"To process: {len(to_process)} records")

    if not to_process:
        print("Nothing to process!")
        return

    print()
    print("=" * 70)
    print(f"  Model:       {args.model}")
    print(f"  Output:      {output_path}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Retries:     {'infinite' if args.retries == 0 else args.retries} (with exponential backoff)")
    print("  Mode:        Forced tool calling + Thinking traces")
    print("=" * 70)
    print()

    # Initialize
    token_stats = TokenStats()
    counters = {"success": 0, "error": 0}
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    # HTTP session with high limits
    connector = aiohttp.TCPConnector(limit=args.concurrency, limit_per_host=args.concurrency)
    timeout = aiohttp.ClientTimeout(total=300)

    mode = "a" if args.resume and output_path.exists() else "w"

    async with (
        aiohttp.ClientSession(connector=connector, timeout=timeout) as session,
        aiofiles.open(output_path, mode) as out_file,
    ):

            # Create tasks
            tasks = [
                process_record(
                    record, args.api_url, args.model,
                    token_stats, session, semaphore, write_lock, out_file, counters,
                    verbose=args.verbose, max_retries=args.retries
                )
                for record in to_process
            ]

            # Progress bar (disabled in verbose mode)
            if not args.verbose:
                pbar = tqdm(
                    total=len(to_process),
                    desc="Rerolling",
                    unit="rec",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                )

            # Process all tasks
            rerolled_results = []
            for coro in asyncio.as_completed(tasks):
                result = await coro
                rerolled_results.append(result)

                if not args.verbose:
                    pbar.update(1)
                    # Update postfix
                    stats = token_stats.get_stats()
                    pbar.set_postfix_str(
                        f"✓{counters['success']} ✗{counters['error']} | "
                        f"{stats['completion_per_sec']:.0f} tok/s | "
                        f"total: {format_tokens(stats['total_tokens'])}"
                    )

            if not args.verbose:
                pbar.close()

    # Write proof file if requested
    if args.proof and rerolled_results:
        # Find first successful result
        for success, uuid, result in rerolled_results:
            if success and isinstance(result, dict):
                original = original_records.get(uuid)
                if original:
                    proof = {
                        "description": "Forced Tool-Call Rerollout Proof",
                        "mode": "Hybrid thinking approach",
                        "model": args.model,
                        "uuid": uuid,
                        "BEFORE": original,
                        "AFTER": result
                    }
                    async with aiofiles.open(args.proof, "w") as proof_file:
                        await proof_file.write(json.dumps(proof, indent=2))
                    print(f"Proof written to {args.proof}")
                    break

    # Verbose comparison output
    if args.verbose and rerolled_results:
        print("\n=== COMPARISON ===")
        for success, uuid, result in rerolled_results[:3]:  # Show first 3
            if success and isinstance(result, dict):
                original = original_records.get(uuid)
                if original:
                    print(f"\nRecord: {uuid[:12]}...")
                    print("BEFORE (assistant turns):")
                    for m in original["messages"]:
                        if m["role"] == "assistant":
                            tc = m.get("tool_calls", [])
                            if tc:
                                print(f"  TOOL: {tc[0]['function']['name']}")
                            else:
                                print(f"  TEXT: \"{m.get('content', '')[:60]}...\"")
                    print("AFTER (assistant turns):")
                    for m in result["messages"]:
                        if m["role"] == "assistant":
                            tc = m.get("tool_calls", [])
                            if tc:
                                print(f"  TOOL: {tc[0]['function']['name']}")
                            else:
                                print(f"  TEXT: \"{m.get('content', '')[:60]}...\"")

    # Final summary
    stats = token_stats.get_stats()
    print()
    print()
    print("=" * 70)
    print("                         COMPLETED")
    print("=" * 70)
    print(f"  Records processed:  {counters['success'] + counters['error']}")
    print(f"  Successful:         {counters['success']}")
    print(f"  Errors:             {counters['error']}")
    print(f"  Time:               {stats['elapsed']/60:.1f} min")
    print(f"  Rate:               {(counters['success'] + counters['error']) / stats['elapsed']:.2f} rec/s")
    print()
    print(f"  Prompt tokens:      {format_tokens(stats['prompt_tokens'])}")
    print(f"  Completion tokens:  {format_tokens(stats['completion_tokens'])}")
    print(f"  Total tokens:       {format_tokens(stats['total_tokens'])}")
    print(f"  Tokens/sec:         {stats['tokens_per_sec']:.0f}")
    print(f"  Completion tok/s:   {stats['completion_per_sec']:.0f}")
    print()
    print(f"  Output:             {output_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Full dataset rerollout with thinking traces")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("-o", "--output", help="Output JSONL file (default: input_rerolled.jsonl)")
    parser.add_argument("-n", "--num", type=int, help="Limit number of records to process")
    parser.add_argument("-i", "--index", type=int, help="Process specific record index only")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    parser.add_argument("--api-url", default="http://localhost:30000/v1/chat/completions")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3.2")
    parser.add_argument("-c", "--concurrency", type=int, default=3000, help="Max concurrent requests")
    parser.add_argument("-r", "--retries", type=int, default=0, help="Max retries per request (0=infinite, default: 0)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--proof", help="Write before/after proof file (first record)")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
