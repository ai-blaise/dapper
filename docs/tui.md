# TUI Application Guide

The TUI provides an interactive way to browse and compare datasets. Supports JSONL, JSON, Parquet, and CSV formats.

> **Recent changes:** The TUI now defaults to a **read-only view mode** — records are displayed as-is without parser_finale processing. Use `-x` to enable export/comparison mode. Keybindings have been centralized and are consistent across all screens. The record list now shows **actual field names** from the data as column headers.

## Quick Start

```bash
# Browse a single file (read-only view)
uv run python -m scripts.tui.app dataset/conversations.jsonl

# Browse with export mode (shows original vs. parsed side-by-side)
uv run python -m scripts.tui.app dataset/conversations.jsonl -x

# Browse a directory
uv run python -m scripts.tui.app dataset/

# Compare two directories
uv run python -m scripts.tui.app dataset_a/ --compare dataset_b/
```

## Supported Formats

| Format | Extensions | Description |
|--------|------------|-------------|
| JSONL | `.jsonl` | One JSON object per line |
| JSON | `.json` | JSON array of objects |
| Parquet | `.parquet`, `.pq` | Apache Parquet columnar format |
| CSV | `.csv` | Comma-separated values |

## Options

| Option | Description |
|--------|-------------|
| `-x, --export` | Enable export mode (comparison view with parser_finale processing). Without this flag, the TUI shows a read-only detail view. |
| `-O, --output-dir` | Output directory for export operations (default: `parsed_datasets`) |
| `-c, --compare` | Path to second dataset for side-by-side comparison |
| `--app-theme` | Set the app theme (CLI takes precedence over config) |
| `--syntax-theme` | Set the syntax highlighting theme for JSON display |

## Themes

The TUI supports two separate theme systems:

### App Theme

Controls the overall look of the TUI (colors, backgrounds, borders). Available themes:

| Theme | Type | Description |
|-------|------|-------------|
| `textual-dark` | Dark | Default Textual dark theme |
| `nord` | Dark | Arctic, north-bluish color palette |
| `gruvbox` | Dark | Retro groove color scheme |
| `tokyo-night` | Dark | Tokyo night color scheme |
| `atom-one-dark` | Dark | Atom One Dark theme |
| `atom-one-light` | Light | Atom One Light theme |
| `solarized-light` | Light | Solarized light theme |
| `solarized-dark` | Dark | Solarized dark theme |

### Syntax Theme

Controls syntax highlighting for JSON display in the field detail modal. Available themes:

| Theme | Description |
|-------|-------------|
| `monokai` | Default - Monokai color scheme |
| `dracula` | Dracula color scheme |
| `nord` | Nord color scheme |
| `gruvbox-dark` | Gruvbox dark color scheme |
| `solarized-dark` | Solarized dark |
| `solarized-light` | Solarized light |

### Configuration

Theme preferences are stored in `config.json` in the project root:

```json
{
  "app_theme": "textual-dark",
  "syntax_theme": "monokai"
}
```

### Keybindings

| Key | Action |
|-----|--------|
| `Ctrl+T` | Cycle through app themes |
| `Ctrl+Y` | Cycle through syntax themes |

### Priority

Theme selection follows this priority order:
1. CLI argument (`--app-theme` / `--syntax-theme`) - highest priority
2. Config file (`config.json`)
3. Default (textual-dark / monokai) - lowest priority

## View Modes

### Read-Only Mode (default)

Browse a dataset and view records as-is — no processing applied:

```bash
uv run python -m scripts.tui.app dataset/conversations.jsonl
uv run python -m scripts.tui.app dataset/
```

**Flow:** File List → Record List → Record Detail View

The record detail view shows the full record as an expandable JSON tree. The record list columns are derived from the **actual field names** in the data (e.g., `conversations`, `agent`, `model`), not hardcoded labels.

### Export Mode (`-x`)

Browse a dataset and view original vs. parsed records side-by-side:

```bash
uv run python -m scripts.tui.app dataset/conversations.jsonl -x
```

**Flow:** File List → Record List → Comparison View

In the comparison view:
- **Left panel**: Original record
- **Right panel**: Parsed output (assistant content emptied, reasoning removed)

This is useful for reviewing what `parser_finale` does to your data. Export keys (`P`, `X`, `x`) are only available in this mode.

### Dataset Comparison Mode

Compare two directories side-by-side (no parsing applied):

```bash
uv run python -m scripts.tui.app left_dataset/ --compare right_dataset/
```

This opens a dual-pane view where each side has independent navigation:

1. **File List** → Select a file to open
2. **Record List** → Select a record to view
3. **JSON View** → See the full record as a tree

Use `h/l` or `Tab` to switch between panes. Each pane navigates independently.

## Keybindings

All keybindings are defined in a single module (`scripts/tui/keybindings.py`) and are consistent across every screen.

### Global (available everywhere)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `m` | Show field detail modal (when a JSON tree is visible) |
| `Ctrl+T` | Cycle app theme |
| `Ctrl+Y` | Cycle syntax theme |

### Navigation

| Key | Action |
|-----|--------|
| `j/k` or `↑/↓` | Move up/down |
| `g` | Jump to top |
| `G` | Jump to bottom |
| `Enter` | Select item / Expand node |
| `ESC` or `b` | Go back |

### Dual-Pane (comparison / dataset comparison mode)

| Key | Action |
|-----|--------|
| `h/l` or `←/→` | Switch pane focus (left/right) |
| `Tab` | Switch pane focus |

### Tree View (record detail / comparison)

| Key | Action |
|-----|--------|
| `e` | Expand all nodes |
| `c` | Collapse all nodes |

### Pagination (record list / dual-pane with large files)

| Key | Action |
|-----|--------|
| `n` | Next page |
| `p` | Previous page |

### Export (requires `-x` mode)

| Key | Screen | Action |
|-----|--------|--------|
| `P` | File List | Export all files in directory |
| `X` | Record List | Export all records in file |
| `x` | Comparison | Export current record |

Exports go to the directory specified by `-O` (default: `parsed_datasets/`).

### Field Detail Modal

| Key | Action |
|-----|--------|
| `ESC` or `Enter` | Close modal |
| `q` | Quit app |

## Schema Detection

The TUI automatically detects field mappings:

- **ID fields**: `uuid`, `id`, `uid`, `example_id`, `trial_name`, `chat_id`, `conversation_id`
- **Message fields**: Arrays with `role` or `content` keys
- **Tool fields**: Arrays with `function` or `name` keys

If no standard schema is detected, the TUI shows raw JSON data. The record list table shows the **actual top-level field names** from the data as column headers, with value previews in each cell.

## Tips

- **Large files**: Files over 100MB use lazy paginated mode (records loaded on demand)
- **Caching**: Records are cached after first load for fast navigation
- **Format detection**: Automatic from file extension
- **Raw field names**: The TUI preserves original field names (e.g., `conversations` for Parquet, `messages` for JSONL) — no normalization is applied by default

## Current Limitations

| Area | Current Behavior | Future Plan |
|------|------------------|-------------|
| Record Matching | Index-based only | ID-based matching |
| Schema Detection | AI conversation patterns | Configurable patterns |
| Panel Labels | Hardcoded | Parameterized |
