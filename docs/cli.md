# Dapper CLI Reference

Dapper installs a global `dapper` command for exploring, viewing, parsing, mixing, and splitting local dataset files from any working directory.

Install it from a checkout:

```bash
uv tool install .
```

Install it from PyPI after publishing:

```bash
uv tool install dapper-datasets
```

`dapper-datasets` is the package name. `dapper` is the terminal command.

After installation, use the same command from any directory:

```bash
dapper --help
```

## Command Coverage

The public `dapper` CLI currently exposes the core Dapper workflows:

| Command | Purpose |
|---------|---------|
| `dapper list` | Print a tabular summary of records |
| `dapper show` | Print a full record or nested field |
| `dapper search` | Search records for text |
| `dapper stats` | Print dataset statistics |
| `dapper view` | Open the interactive TUI |
| `dapper parse` | Empty assistant responses while preserving prompt/tool structure |
| `dapper mix` | Mix supported dataset directories into unified Parquet output |
| `dapper split` | Split JSONL or Parquet files into parts |

Some repository scripts are still internal or legacy and do not yet have public `dapper` wrappers. This includes the rerollout scripts, upload helpers, filtering helpers, and demo scripts under `scripts/`.

## Global Help

```bash
dapper --help
```

Shows the available top-level commands. Use `--help` after a command to inspect its flags:

```bash
dapper parse --help
dapper mix --help
```

## `dapper list`

Print a compact table for records in a JSONL, JSON, or Parquet file.

```bash
dapper list <file> [options]
```

Options:

| Option | Description |
|--------|-------------|
| `-n, --limit N` | Limit output to N displayed records |
| `--has-tools` | Only show records with tool definitions |
| `--has-reasoning` | Only show records with reasoning content |
| `--min-messages N` | Only show records with at least N messages |
| `--input-format FORMAT` | Input format: `auto`, `jsonl`, `json`, `parquet` |

Examples:

```bash
dapper list dataset/conversations.jsonl -n 10
dapper list dataset/conversations.parquet --has-tools
dapper list dataset/conversations.jsonl --min-messages 5
dapper list dataset/conversations.jsonl --has-tools --has-reasoning -n 20
```

## `dapper show`

Print one record by zero-based index, or extract a specific nested field.

```bash
dapper show <file> <index> [options]
```

Options:

| Option | Description |
|--------|-------------|
| `-f, --field PATH` | Extract a nested field using dot/bracket notation |
| `--input-format FORMAT` | Input format: `auto`, `jsonl`, `json`, `parquet` |

Field path examples:

| Path | Result |
|------|--------|
| `messages` | Full messages array |
| `messages[0]` | First message |
| `messages[0].content` | First message content |
| `tools[1].function.name` | Name of the second tool definition |

Examples:

```bash
dapper show dataset/conversations.jsonl 0
dapper show dataset/conversations.jsonl 0 -f messages
dapper show dataset/conversations.jsonl 0 -f messages[1].content
dapper show dataset/conversations.jsonl 5 -f uuid
dapper show dataset/conversations.jsonl 0 -f tools
```

## `dapper search`

Search records for text.

```bash
dapper search <file> <query> [options]
```

Options:

| Option | Description |
|--------|-------------|
| `-n, --limit N` | Limit number of matches, default `20` |
| `-c, --context` | Show nearby matching text |
| `--case-sensitive` | Use case-sensitive matching |
| `--input-format FORMAT` | Input format: `auto`, `jsonl`, `json`, `parquet` |

Examples:

```bash
dapper search dataset/conversations.jsonl "API"
dapper search dataset/conversations.jsonl "error" -c
dapper search dataset/conversations.jsonl "API" --case-sensitive
dapper search dataset/conversations.jsonl "function" -n 5
```

## `dapper stats`

Print dataset statistics.

```bash
dapper stats <file> [options]
```

Options:

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Include additional tool-name details |
| `--input-format FORMAT` | Input format: `auto`, `jsonl`, `json`, `parquet` |

Examples:

```bash
dapper stats dataset/conversations.jsonl
dapper stats dataset/conversations.parquet -v
```

## `dapper view`

Open the interactive terminal UI for a file or directory.

```bash
dapper view <path> [options]
```

Options:

| Option | Description |
|--------|-------------|
| `-O, --output-dir DIR` | Output directory for export operations |
| `-c, --compare PATH` | Compare against a second file or directory |
| `-x, --export` | Enable parser/export comparison mode |
| `--app-theme THEME` | Textual app theme |
| `--syntax-theme THEME` | Syntax highlighting theme |

Examples:

```bash
dapper view dataset/conversations.jsonl
dapper view dataset/conversations.parquet
dapper view dataset/
dapper view dataset_a/ --compare dataset_b/
dapper view dataset/conversations.jsonl -x
```

## `dapper parse`

Process records by emptying assistant message content while preserving system/user/tool messages, tool calls, metadata, and conversation structure.

```bash
dapper parse <path> [options]
```

File mode writes to stdout by default. Use `-o` for a specific file or `-O` for generated output in a directory.

Options:

| Option | Description |
|--------|-------------|
| `--input-format FORMAT` | Input format: `auto`, `jsonl`, `json`, `parquet` |
| `-f, --format, --output-format FORMAT` | Output format: `json`, `jsonl`, `parquet`, `markdown`, `text` |
| `-o, --output FILE` | Output file path, default stdout |
| `-O, --output-dir DIR` | Output directory for generated `{stem}_parsed.{format}` files |
| `-i, --index N` | Process only one record |
| `--start N` | Start index for range processing |
| `--end N` | End index for range processing |
| `--has-tools` | Only include records with tools |
| `--compact` | Compact JSON output |

Examples:

```bash
dapper parse dataset/train.jsonl
dapper parse dataset/train.jsonl -f jsonl -o prompts.jsonl
dapper parse dataset/train.parquet -f jsonl -o prompts.jsonl
dapper parse dataset/train.jsonl -i 5 -f markdown
dapper parse dataset/train.jsonl --start 0 --end 100 -o sample.json
dapper parse dataset/train.jsonl --has-tools -f json -o tools_only.json
dapper parse dataset/train.jsonl -O parsed_datasets/
```

Parquet output requires a file destination:

```bash
dapper parse dataset/train.jsonl -f parquet -o output.parquet
```

## `dapper mix`

Mix supported dataset directories into unified Parquet training data.

```bash
dapper mix <input_dir> [options]
```

Options:

| Option | Description |
|--------|-------------|
| `-o, --output PATH` | Output Parquet file path, default `mixed_output.parquet` |
| `--dry-run` | Show record counts without writing output |
| `--include [SOURCE ...]` | Only include matching source dataset prefixes |
| `--exclude [SOURCE ...]` | Exclude matching source dataset prefixes |
| `--batch-size N` | Records per write batch, default `500` |
| `--tooling-sample-rate RATE` | Sample `Nemotron-SFT-Agentic-v2` tool-calling records only |
| `--sample-seed SEED` | Seed for tool-calling sampling |
| `--resume` | Resume from existing output when possible |
| `--shuffle` | Shuffle records before writing |
| `--shuffle-seed SEED` | Seed for shuffle |
| `--num-chunks N` | Split output into N Parquet chunks |

Examples:

```bash
dapper mix datasets/ --dry-run
dapper mix datasets/ -o output.parquet
dapper mix datasets/ -o nemotron.parquet --include Nemotron
dapper mix datasets/ -o agentic.parquet --include Nemotron-SFT-Agentic-v2
dapper mix datasets/ -o non_nemotron.parquet --exclude Nemotron
```

Sampling examples:

```bash
dapper mix datasets/ -o agentic_50.parquet \
  --include Nemotron-SFT-Agentic-v2 \
  --tooling-sample-rate 0.5 \
  --sample-seed 42
```

Shuffle and chunk examples:

```bash
dapper mix datasets/ -o shuffled.parquet \
  --include Nemotron-SFT-Agentic-v2 \
  --shuffle --shuffle-seed 42

dapper mix datasets/ -o chunks/distill \
  --include Nemotron-SFT-Agentic-v2 \
  --shuffle --shuffle-seed 42 \
  --num-chunks 8
```

## `dapper split`

Split JSONL or Parquet files into N parts.

```bash
dapper split <input_file> -n <parts> [options]
```

Options:

| Option | Description |
|--------|-------------|
| `-n, --parts N` | Number of parts to create |
| `-o, --output-dir DIR` | Output directory, default same as input |
| `--prefix PREFIX` | Output filename prefix, default input filename |
| `--dry-run` | Show split plan without writing files |
| `--verify` | Verify parts recombine to the original file |
| `--shuffle` | Shuffle records before splitting |
| `--shuffle-seed SEED` | Seed for reproducible shuffling |

Examples:

```bash
dapper split dataset/train.jsonl -n 4
dapper split dataset/train.jsonl -n 10 --dry-run
dapper split dataset/train.jsonl -n 5 --output-dir ./splits/ --prefix training_data
dapper split dataset/train.jsonl -n 4 --verify
dapper split dataset/train.jsonl -n 4 --shuffle --shuffle-seed 42
dapper split mixed.parquet -n 8 --shuffle --shuffle-seed 42
```

## Developer Notes

The implementation modules under `scripts/` still exist for development and backwards compatibility, but user-facing docs and workflows should prefer `dapper`.
