# Dapper Usage

Dapper provides a global `dapper` command for local dataset inspection, parsing, comparison, mixing, and splitting.

Install from a checkout:

```bash
uv tool install .
```

Run from a development checkout:

```bash
dapper --help
```

## Record Structure

Common AI conversation records contain these fields:

| Field | Description |
|-------|-------------|
| `uuid` | Unique identifier for the record |
| `messages` | Conversation messages with roles such as system, user, assistant, and tool |
| `conversations` | Parquet-oriented conversation field used by mixed outputs |
| `tools` | Tool/function definitions available to the assistant |
| `license` | License type |
| `used_in` | Dataset/model usage tags |
| `reasoning` | Reasoning mode flag, when present |

## Interactive TUI

Browse a file or directory:

```bash
dapper view dataset/interactive_agent.jsonl
dapper view dataset/
dapper view dataset_a/ --compare dataset_b/
```

Useful flags:

| Flag | Description |
|------|-------------|
| `-x, --export` | Show original vs. parsed records and enable export keys |
| `-O, --output-dir DIR` | Directory for TUI exports |
| `--compare PATH` | Side-by-side dataset comparison |
| `--app-theme THEME` | Textual app theme |
| `--syntax-theme THEME` | JSON syntax theme |

## Parse Records

`dapper parse` preserves conversation structure but empties assistant response text. Tool calls are preserved.

```bash
# JSON output to stdout
dapper parse dataset/interactive_agent.jsonl

# Specific record by index
dapper parse dataset/interactive_agent.jsonl -i 5

# Range of records
dapper parse dataset/interactive_agent.jsonl --start 0 --end 10

# Different output formats
dapper parse dataset/interactive_agent.jsonl -f json
dapper parse dataset/interactive_agent.jsonl -f jsonl
dapper parse dataset/interactive_agent.jsonl -f markdown
dapper parse dataset/interactive_agent.jsonl -f text

# Output to file
dapper parse dataset/interactive_agent.jsonl -o output.json

# Filter records with tools only
dapper parse dataset/interactive_agent.jsonl --has-tools

# Compact JSON
dapper parse dataset/interactive_agent.jsonl --compact
```

Parse options:

| Option | Description |
|--------|-------------|
| `--input-format FORMAT` | Input format: `auto`, `jsonl`, `json`, `parquet` |
| `-f, --format FORMAT` | Output format: `json`, `jsonl`, `parquet`, `markdown`, `text` |
| `-o, --output FILE` | Output file path, default stdout |
| `-O, --output-dir DIR` | Generated output directory |
| `-i, --index N` | Process one record |
| `--start N` | Start index |
| `--end N` | End index |
| `--has-tools` | Only include records with tool definitions |
| `--compact` | Compact JSON output |

## Explore Records

| Command | Description | Example |
|---------|-------------|---------|
| `dapper list` | Tabular summary of records | `dapper list dataset/file.jsonl -n 10` |
| `dapper show` | View record or specific field | `dapper show dataset/file.jsonl 0 -f messages[1]` |
| `dapper search` | Find text in records | `dapper search dataset/file.jsonl "query" -c` |
| `dapper stats` | Dataset statistics | `dapper stats dataset/file.jsonl -v` |

Examples:

```bash
dapper list dataset/interactive_agent.jsonl -n 10
dapper show dataset/interactive_agent.jsonl 0 -f messages[1]
dapper search dataset/interactive_agent.jsonl "Bitcoin" -c
dapper stats dataset/interactive_agent.jsonl -v
```

## Mix Datasets

Combine supported dataset directories into unified Parquet output:

```bash
dapper mix datasets/ --dry-run
dapper mix datasets/ -o output.parquet
dapper mix datasets/ -o nemotron.parquet --include Nemotron
dapper mix datasets/ -o agentic_50.parquet \
  --include Nemotron-SFT-Agentic-v2 \
  --tooling-sample-rate 0.5 \
  --sample-seed 42
```

## Split Datasets

Split JSONL or Parquet files into parts:

```bash
dapper split dataset/file.jsonl -n 4
dapper split dataset/file.jsonl -n 10 --dry-run
dapper split mixed.parquet -n 8 --shuffle --shuffle-seed 42
```

## Command Coverage

The public `dapper` CLI currently covers the core workflows: `list`, `show`, `search`, `stats`, `view`, `parse`, `mix`, and `split`. Other scripts in this directory are internal or legacy until explicit `dapper` wrappers are added.
