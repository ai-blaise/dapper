# Plan: Refactor to Dapper `src/` Layout

## Overview

Refactor `/home/archimedes/code/blaise/dataset-parser` into a properly named, pip-installable Python package named **dapper** (Dataset Absurdly Powerful Parser of Epic Robustness). The project uses the modern `src/` layout where `pip install -e .` installs from `src/dapper/`.

## Target Directory Structure

```
dapper/                              # repo root = package name (renamed from dataset-parser)
├── src/dapper/                      # pip installs from here
│   ├── __init__.py
│   ├── __main__.py
│   ├── scripts/
│   │   ├── __init__.py
│   │   ├── main.py                  # list, show, search, stats
│   │   ├── parser_finale.py         # parse command
│   │   ├── data_splitter.py         # split command
│   │   ├── filter_evals.py
│   │   ├── upload_to_hf.py
│   │   ├── data_formats/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── csv_loader.py
│   │   │   ├── jsonl_loader.py
│   │   │   ├── json_loader.py
│   │   │   ├── parquet_loader.py
│   │   │   ├── format_detector.py
│   │   │   ├── schema_normalizer.py
│   │   │   └── directory_loader.py
│   │   ├── dataset_mixer/
│   │   │   ├── __init__.py
│   │   │   ├── __main__.py
│   │   │   ├── cli.py
│   │   │   ├── mixer.py
│   │   │   ├── adapters.py
│   │   │   └── schema.py
│   │   └── tui/
│   │       ├── __init__.py
│   │       ├── app.py               # view command
│   │       ├── data_loader.py
│   │       ├── keybindings.py
│   │       ├── views/
│   │       ├── mixins/
│   │       ├── widgets/
│   │       ├── screens/
│   │       └── styles/
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       └── streaming.py
├── rerollout_scripts/               # dev-only, NOT installed
│   ├── rerollout_simple.py
│   ├── rerollout_full.py
│   ├── rerollout_forced.py
│   └── rerollout_curl_example.sh
├── tests/
├── docs/
├── plans/
├── pyproject.toml
├── README.md
└── uv.lock
```

## Step-by-Step Plan

### Phase 1: Preparatory

### Step 1.1: Create `rerollout_scripts/` and Move Rerollout Scripts

**What:** Create `rerollout_scripts/` directory and move all rerollout scripts out of `scripts/`.
**Why:** These are dev-only helper scripts for LLM regeneration operations, not part of the dapper package. They must be excluded from the pip install.
**Source:** `scripts/rerollout*.py`, `scripts/rerollout*.sh` — confirmed by examining `scripts/` directory.

**Files to move:**
- `scripts/rerollout_simple.py` → `rerollout_scripts/rerollout_simple.py`
- `scripts/rerollout_full.py` → `rerollout_scripts/rerollout_full.py`
- `scripts/rerollout_forced.py` → `rerollout_scripts/rerollout_forced.py`
- `scripts/rerollout_curl_example.sh` → `rerollout_scripts/rerollout_curl_example.sh`

### Step 1.2: Create `src/` Directory Structure

**What:** Create the `src/dapper/` directory and its subdirectories.
**Why:** The `src/` layout is the modern Python packaging standard. pip installs from `src/dapper/` as the `dapper` package.
**Source:** [Python Packaging Authority — src layout](https://packaging.python.org/en/latest/tutorials/packaging-projects/#configuring-metadata)

**Directories to create:**
- `src/dapper/`
- `src/dapper/scripts/`
- `src/dapper/scripts/data_formats/`
- `src/dapper/scripts/dataset_mixer/`
- `src/dapper/scripts/tui/`
- `src/dapper/scripts/tui/views/`
- `src/dapper/scripts/tui/mixins/`
- `src/dapper/scripts/tui/widgets/`
- `src/dapper/scripts/tui/screens/`
- `src/dapper/scripts/tui/styles/`
- `src/dapper/utils/`

### Step 1.3: Create `src/dapper/__init__.py`

**What:** Create `src/dapper/__init__.py` with package metadata.
**Why:** Required for pip to recognize `src/dapper/` as a package.
**Source:** Standard Python package init.

```python
"""dapper — Dataset Absurdly Powerful Parser of Epic Robustness."""

__version__ = "0.1.0"
```

### Step 1.4: Create `src/dapper/__main__.py`

**What:** Create `src/dapper/__main__.py` to allow `python -m dapper`.
**Why:** Enables `uv run python -m dapper` and `python -m dapper` entry point.
**Source:** Standard Python `__main__` pattern.

```python
"""Allow running dapper as a module: python -m dapper"""
from dapper.scripts.main import main

if __name__ == "__main__":
    main()
```

### Phase 2: Move Source Files

### Step 2.1: Move `scripts/` → `src/dapper/scripts/`

**What:** Move all files from `scripts/` (except rerollout scripts already moved) into `src/dapper/scripts/`.
**Why:** The `scripts/` directory contains the main application code that becomes part of the `dapper` package.
**Source:** Target structure defined in Overview.

**Files to move:**
- `scripts/main.py` → `src/dapper/scripts/main.py`
- `scripts/parser_finale.py` → `src/dapper/scripts/parser_finale.py`
- `scripts/data_splitter.py` → `src/dapper/scripts/data_splitter.py`
- `scripts/filter_evals.py` → `src/dapper/scripts/filter_evals.py`
- `scripts/upload_to_hf.py` → `src/dapper/scripts/upload_to_hf.py`
- `scripts/usage.md` → `src/dapper/scripts/usage.md`
- `scripts/data_formats/` → `src/dapper/scripts/data_formats/`
- `scripts/dataset_mixer/` → `src/dapper/scripts/dataset_mixer/`
- `scripts/tui/` → `src/dapper/scripts/tui/`

### Step 2.2: Move `utils/` → `src/dapper/utils/`

**What:** Move `utils/` directory into `src/dapper/utils/`.
**Why:** The `utils/` package is used by `scripts/` modules, so it moves inside the `dapper/` package to maintain proper import relationships.
**Source:** `utils/__init__.py`, `utils/data.py`, `utils/streaming.py`, `utils/config.py` — all used by scripts modules.

**Files to move:**
- `utils/__init__.py` → `src/dapper/utils/__init__.py`
- `utils/config.py` → `src/dapper/utils/config.py`
- `utils/data.py` → `src/dapper/utils/data.py`
- `utils/streaming.py` → `src/dapper/utils/streaming.py`

### Step 2.3: Create Package Init Files

**What:** Create `src/dapper/scripts/__init__.py` and `src/dapper/scripts/tui/__init__.py`.
**Why:** These subdirectories need `__init__.py` files to be proper subpackages.
**Source:** Standard Python package requirement.

### Phase 3: Update Configuration

### Step 3.1: Update `pyproject.toml`

**What:** Rewrite `pyproject.toml` with the new package name, `src/` layout configuration, and CLI entry points.
**Why:** The package needs proper pip configuration with entry points so `pip install -e .` creates the `dapper` command.
**Source:** Current `pyproject.toml` with `name = "dataset-parser"`; [tool.setup.packages.find] configuration.

**Changes to make:**
```toml
[project]
name = "dapper"
version = "0.1.0"
description = "Dataset Absurdly Powerful Parser of Epic Robustness — A toolkit for exploring and transforming AI conversation datasets"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "textual>=7.3.0",
    "pyarrow>=15.0.0",
    "huggingface-hub>=1.3.2",
    "requests>=2.32.5",
    "aiohttp>=3.13.3",
    "aiofiles>=25.1.0",
]

[project.scripts]
dapper = "dapper.scripts.main:main"
dapper-list = "dapper.scripts.main:cmd_list"
dapper-show = "dapper.scripts.main:cmd_show"
dapper-search = "dapper.scripts.main:cmd_search"
dapper-stats = "dapper.scripts.main:cmd_stats"
dapper-parse = "dapper.scripts.parser_finale:main"
dapper-split = "dapper.scripts.data_splitter:main"
dapper-mix = "dapper.scripts.dataset_mixer.cli:main"
dapper-view = "dapper.scripts.tui.app:main"

[tool.setup.packages.find]
where = ["src"]
include = ["dapper*"]
exclude = ["rerollout_scripts*", "tests*", "docs*", "plans*"]

[tool.setup.package-data]
dapper = ["py.typed"]
```

### Phase 4: Update Import Paths

### Step 4.1: Update All Python Import Paths

**What:** Update every file that imports from `scripts.*`, `utils.*`, or has internal imports to use `dapper.scripts.*`, `dapper.utils.*`.
**Why:** After restructuring, all import paths must reflect the new package location.
**Source:** Found 165+ occurrences of `from scripts.` or `import scripts.` across the codebase.

**Files with imports to update (comprehensive list):**

#### `src/dapper/scripts/main.py`
- `from scripts.data_formats import get_loader, get_loader_for_format, normalize_record` → `from dapper.scripts.data_formats import get_loader, get_loader_for_format, normalize_record`

#### `src/dapper/scripts/parser_finale.py`
- `from scripts.data_formats import get_loader, get_loader_for_format, normalize_record` → `from dapper.scripts.data_formats import get_loader, get_loader_for_format, normalize_record`
- `from scripts.tui.app import JsonComparisonApp` → `from dapper.scripts.tui.app import JsonComparisonApp`

#### `src/dapper/scripts/data_splitter.py`
- No `scripts.*` imports (uses only standard library + pathlib)

#### `src/dapper/scripts/filter_evals.py`
- No `scripts.*` imports

#### `src/dapper/scripts/upload_to_hf.py`
- No `scripts.*` imports

#### `src/dapper/scripts/data_formats/__init__.py`
- All `from scripts.data_formats.*` → `from dapper.scripts.data_formats.*`
- All `from scripts.*` → `from dapper.scripts.*`

#### `src/dapper/scripts/data_formats/base.py`
- All `from scripts.data_formats.*` → `from dapper.scripts.data_formats.*`

#### `src/dapper/scripts/data_formats/csv_loader.py`
- `from scripts.data_formats.base import DataLoader` → `from dapper.scripts.data_formats.base import DataLoader`

#### `src/dapper/scripts/data_formats/jsonl_loader.py`
- `from scripts.data_formats.base import DataLoader` → `from dapper.scripts.data_formats.base import DataLoader`

#### `src/dapper/scripts/data_formats/json_loader.py`
- `from scripts.data_formats.base import DataLoader` → `from dapper.scripts.data_formats.base import DataLoader`

#### `src/dapper/scripts/data_formats/parquet_loader.py`
- `from scripts.data_formats.base import DataLoader` → `from dapper.scripts.data_formats.base import DataLoader`

#### `src/dapper/scripts/data_formats/format_detector.py`
- `from scripts.data_formats.base import DataLoader` → `from dapper.scripts.data_formats.base import DataLoader`

#### `src/dapper/scripts/data_formats/schema_normalizer.py`
- All `from scripts.data_formats.*` → `from dapper.scripts.data_formats.*`

#### `src/dapper/scripts/data_formats/directory_loader.py`
- All `from scripts.data_formats.*` → `from dapper.scripts.data_formats.*`

#### `src/dapper/scripts/dataset_mixer/__main__.py`
- `from scripts.dataset_mixer.cli import main` → `from dapper.scripts.dataset_mixer.cli import main`

#### `src/dapper/scripts/dataset_mixer/cli.py`
- `from scripts.dataset_mixer.mixer import mix` → `from dapper.scripts.dataset_mixer.mixer import mix`

#### `src/dapper/scripts/dataset_mixer/mixer.py`
- `from scripts.data_formats.format_detector import EXTENSION_MAP` → `from dapper.scripts.data_formats.format_detector import EXTENSION_MAP`
- `from scripts.dataset_mixer.adapters import detect_adapter` → `from dapper.scripts.dataset_mixer.adapters import detect_adapter`
- `from scripts.dataset_mixer.schema import OUTPUT_SCHEMA` → `from dapper.scripts.dataset_mixer.schema import OUTPUT_SCHEMA`
- `from utils import get_existing_record_count, stream_file` → `from dapper.utils import get_existing_record_count, stream_file`

#### `src/dapper/scripts/dataset_mixer/adapters.py`
- `from scripts.data_formats.csv_loader import CSVLoader` → `from dapper.scripts.data_formats.csv_loader import CSVLoader`
- `from scripts.data_formats.format_detector import detect_format, get_loader` → `from dapper.scripts.data_formats.format_detector import detect_format, get_loader`
- `from scripts.data_formats.jsonl_loader import JSONLLoader` → `from dapper.scripts.data_formats.jsonl_loader import JSONLLoader`
- `from scripts.data_formats.parquet_loader import ParquetLoader` → `from dapper.scripts.data_formats.parquet_loader import ParquetLoader`
- `from scripts.dataset_mixer.schema import OUTPUT_SCHEMA` → `from dapper.scripts.dataset_mixer.schema import OUTPUT_SCHEMA`

#### `src/dapper/scripts/dataset_mixer/schema.py`
- No `scripts.*` imports (uses only pyarrow)

#### `src/dapper/scripts/tui/app.py`
- `from scripts.data_formats import detect_format, discover_data_files` → `from dapper.scripts.data_formats import detect_format, discover_data_files`
- `from scripts.tui.data_loader import ...` → `from dapper.scripts.tui.data_loader import ...`
- `from scripts.tui.keybindings import GLOBAL_BINDINGS` → `from dapper.scripts.tui.keybindings import GLOBAL_BINDINGS`
- `from scripts.tui.mixins import BackgroundTaskMixin` → `from dapper.scripts.tui.mixins import BackgroundTaskMixin`
- `from scripts.tui.screens import ExportingScreen, LoadingScreen` → `from dapper.scripts.tui.screens import ExportingScreen, LoadingScreen`
- `from scripts.tui.views.*` → `from dapper.scripts.tui.views.*`
- `from scripts.tui.widgets.*` → `from dapper.scripts.tui.widgets.*`
- `from utils.config import get_app_theme, get_syntax_theme` → `from dapper.utils.config import get_app_theme, get_syntax_theme`
- `from scripts.tui.views.dual_record_list_screen import DualRecordListScreen` → `from dapper.scripts.tui.views.dual_record_list_screen import DualRecordListScreen`

#### `src/dapper/scripts/tui/data_loader.py`
- `from scripts.data_formats import get_loader, normalize_record` → `from dapper.scripts.data_formats import get_loader, normalize_record`
- `from scripts.parser_finale import process_record` → `from dapper.scripts.parser_finale import process_record`
- `from scripts.data_formats.parquet_loader import ParquetLoader` → `from dapper.scripts.data_formats.parquet_loader import ParquetLoader`

#### `src/dapper/scripts/tui/keybindings.py`
- No `scripts.*` imports

#### `src/dapper/scripts/tui/__init__.py`
- `from scripts.tui.app import JsonComparisonApp` → `from dapper.scripts.tui.app import JsonComparisonApp`

#### `src/dapper/scripts/tui/views/__init__.py`
- `from scripts.tui.views.*` → `from dapper.scripts.tui.views.*`

#### `src/dapper/scripts/tui/views/*.py`
- All files in `views/` with `from scripts.tui.*` → `from dapper.scripts.tui.*`

#### `src/dapper/scripts/tui/mixins/__init__.py`
- `from scripts.tui.mixins.*` → `from dapper.scripts.tui.mixins.*`

#### `src/dapper/scripts/tui/mixins/*.py`
- All files in `mixins/` with `from scripts.tui.*` → `from dapper.scripts.tui.*`
- `from scripts.tui.keybindings` → `from dapper.scripts.tui.keybindings`

#### `src/dapper/scripts/tui/widgets/__init__.py`
- `from scripts.tui.widgets.*` → `from dapper.scripts.tui.widgets.*`

#### `src/dapper/scripts/tui/widgets/*.py`
- All files in `widgets/` with `from scripts.tui.*` → `from dapper.scripts.tui.*`

#### `src/dapper/scripts/tui/screens/__init__.py`
- `from scripts.tui.screens.*` → `from dapper.scripts.tui.screens.*`

#### `src/dapper/scripts/tui/screens/*.py`
- All files in `screens/` with `from scripts.tui.*` → `from dapper.scripts.tui.*`

#### `src/dapper/utils/data.py`
- `from scripts.dataset_mixer.schema import OUTPUT_SCHEMA` → `from dapper.scripts.dataset_mixer.schema import OUTPUT_SCHEMA`

#### `src/dapper/utils/streaming.py`
- `from scripts.data_formats import detect_format, get_loader` → `from dapper.scripts.data_formats import detect_format, get_loader`
- `from scripts.dataset_mixer.adapters import BaseAdapter, detect_adapter` → `from dapper.scripts.dataset_mixer.adapters import BaseAdapter, detect_adapter`
- `from scripts.dataset_mixer.schema import OUTPUT_SCHEMA, TURN_TYPE` → `from dapper.scripts.dataset_mixer.schema import OUTPUT_SCHEMA, TURN_TYPE`
- `from .data import transform_batch` → `from .data import transform_batch` (already relative)

#### `src/dapper/utils/config.py`
- No `scripts.*` imports

### Phase 5: Update Documentation

### Step 5.1: Update `README.md`

**What:** Rewrite README.md for the dapper package.
**Why:** The README currently references "dataset-parser" and old CLI invocation patterns.
**Source:** `README.md:1-311`

**Key changes:**
- Title: `D.A.P.P.E.R. — Dataset Absurdly Powerful Parser of Epic Robustness`
- Installation: `pip install dapper` or `uv pip install dapper`
- CLI examples:
  - `dapper list <file>` (was `python -m scripts.main list`)
  - `dapper view <path>` (was `python -m scripts.tui.app`)
  - `dapper parse <path>` (was `python -m scripts.parser_finale`)
  - `dapper split <file> -n 4` (was `python -m scripts.data_splitter`)
  - `dapper mix <dir>` (was `python -m scripts.dataset_mixer`)
- Project structure: update paths from `scripts/` to `src/dapper/scripts/`
- Add Homebrew installation section once Plan 2 (Homebrew) is complete

### Step 5.2: Update `docs/architecture.md`

**What:** Update the architecture doc to reflect the new directory structure and entry points.
**Why:** Documentation must match the new package structure.
**Source:** `docs/architecture.md`

**Key changes:**
- Line 25: `dataset-parser/` → `dapper/` directory structure
- Lines 89-90, 169-170: "dataset-parser Application" → "dapper Application"
- Line 119-124: Entry Points table — `scripts.` → `dapper.scripts.`
- All `scripts/` path references → `src/dapper/scripts/`
- Lines 347-352: Dependencies section

### Step 5.3: Update `docs/cli.md`

**What:** Update CLI documentation with new command names.
**Why:** All `uv run python -m scripts.*` → `dapper *` commands.
**Source:** `docs/cli.md`

### Step 5.4: Rename `docs/tui.md` → `docs/view.md`

**What:** Rename the TUI doc to `view.md` to match the `dapper view` command rename.
**Why:** Consistency — the command is now `dapper view`, not `dapper tui`.
**Source:** CLI rename requirement.

**Files to update:**
- Rename `docs/tui.md` → `docs/view.md`
- Update all references from `dapper tui` → `dapper view` and `python -m scripts.tui.app` → `dapper view`

### Step 5.5: Update `docs/parser-finale.md`

**What:** Update all `scripts.parser_finale` references to `dapper parse`.
**Why:** CLI command rename.
**Source:** `docs/parser-finale.md`

### Step 5.6: Update `docs/data-splitter.md`

**What:** Update all `scripts.data_splitter` references to `dapper split`.
**Why:** CLI command rename.
**Source:** `docs/data-splitter.md`

### Step 5.7: Update `docs/data-formats.md`

**What:** Update all `scripts.data_formats` references to `dapper.scripts.data_formats`.
**Why:** Import path change.
**Source:** `docs/data-formats.md`

### Step 5.8: Update `docs/verify-datasets.md`

**What:** Update any remaining references to old import paths or CLI commands.
**Why:** Consistency.
**Source:** `docs/verify-datasets.md`

### Step 5.9: Update `docs/record-structure.md`

**What:** Update any remaining references to old import paths or CLI commands.
**Why:** Consistency.
**Source:** `docs/record-structure.md`

### Step 5.10: Update All Plan Files

**What:** Update all files in `plans/` to reference the new package name and import paths.
**Why:** Plans are part of the project documentation.
**Source:** All files in `plans/active-plans/` and `plans/graveyard_of_plans/`

### Phase 6: Update Tests

### Step 6.1: Update `tests/conftest.py`

**What:** Review and update test fixtures and imports if needed.
**Why:** Tests must work with the new package structure.
**Source:** `tests/conftest.py`

### Step 6.2: Update All Test Files

**What:** Update all test files with `from scripts.*` or `import scripts.*` imports.
**Why:** Import paths have changed.
**Source:** All files in `tests/` directory.

**Files likely to need updates:**
- `tests/test_cli.py` — `PARSER_MODULE = "scripts.parser_finale"` → `"dapper.scripts.parser_finale"`
- `tests/test_dataset_mixer.py` — all `from scripts.*` → `from dapper.scripts.*`
- `tests/test_multiformat_tui.py` — all `from scripts.tui.*` → `from dapper.scripts.tui.*`
- `tests/test_csv_loader.py` — all `from scripts.data_formats` → `from dapper.scripts.data_formats`
- All other test files with `scripts.*` imports

### Phase 7: Cleanup

### Step 7.1: Remove `scripts/` Directory

**What:** Remove the now-empty `scripts/` directory from the repo root.
**Why:** All source code has moved to `src/dapper/scripts/`.
**Source:** Move completed in Step 2.1.

### Step 7.2: Remove `utils/` Directory

**What:** Remove the now-empty `utils/` directory from the repo root.
**Why:** All utility code has moved to `src/dapper/utils/`.
**Source:** Move completed in Step 2.2.

### Step 7.3: Remove Root `main.py`

**What:** Remove the stub `main.py` at repo root.
**Why:** The stub just printed a greeting. The real entry point is now `src/dapper/__main__.py`.
**Source:** `main.py:1-6` — confirmed stub content.

### Step 7.4: Regenerate `uv.lock`

**What:** Regenerate the lockfile to reflect the new package structure.
**Why:** `uv.lock` tracks dependencies for `src/dapper/` package.
**Source:** Standard `uv` practice after restructuring.

```bash
uv lock
```

### Step 7.5: Exclude `rerollout_scripts/` from pip

**What:** Ensure `rerollout_scripts/` is excluded from the pip package.
**Why:** These are dev-only scripts, not part of the dapper package.
**Source:** `pyproject.toml` `[tool.setup.packages.find]` `exclude = ["rerollout_scripts*"]`

## Verification

After completing all refactoring steps, run the following commands to verify:

```bash
# 1. Navigate to the repo
cd /home/archimedes/code/blaise/dataset-parser

# 2. Reinstall the package in development mode
uv pip install -e .

# 3. Verify the package imports
python -c "import dapper; print(dapper.__version__)"
python -c "from dapper.scripts.main import main; print('main OK')"
python -c "from dapper.scripts.parser_finale import main; print('parser_finale OK')"
python -c "from dapper.scripts.data_splitter import main; print('data_splitter OK')"
python -c "from dapper.scripts.dataset_mixer.cli import main; print('mixer OK')"
python -c "from dapper.scripts.tui.app import main; print('tui OK')"
python -c "from dapper.utils import get_existing_record_count, stream_file; print('utils OK')"

# 4. Verify CLI entry points work
dapper --help
dapper list --help
dapper view --help
dapper parse --help
dapper split --help
dapper mix --help

# 5. Run the test suite
uv run pytest tests/ -v

# 6. Verify specific functionality
dapper list tests/fixtures/valid/minimal.jsonl
dapper parse tests/fixtures/valid/minimal.jsonl --format jsonl
```

## Risks / Clarifications

### 1. Python Version

@architect: pyproject.toml currently uses `requires-python = ">=3.12"`. Should this stay as `>=3.12` or be updated to match Homebrew's `python@3.14`?

| Option | Description |
|--------|-------------|
| `>=3.12` | Current value, more compatible with older systems |
| `>=3.14` | Matches Homebrew default, slightly less compatible |

### 2. TUI Command Rename Alias

@architect: Should `dapper tui` work as an alias for `dapper view` for backward compatibility? Or is `dapper view` a clean break?

### 3. config.json Location

@architect: The `config.json` file (used by `dapper/utils/config.py` for theme preferences) is currently at the repo root. Should it:
- A) Move into the package at `src/dapper/config.json`
- B) Stay at repo root (user configuration)
- C) Move to user home `~/.config/dapper/config.json`

### 4. scripts/usage.md Status

@architect: The `scripts/usage.md` file is a quick reference guide. Should it:
- A) Be updated and kept as `src/dapper/scripts/usage.md`
- B) Merged into README.md
- C) Removed (docs/ is comprehensive)

### 5. Test File Imports

@architect: Test files may have hardcoded paths to fixtures or sample data. Should we verify fixture paths after the move?

## Execution

@minimax-m2: Once the architect approves, execute the steps in order. Start with Phase 1 (Preparatory), then Phase 2 (Move Source Files), then Phase 3 (Update Configuration), then Phase 4 (Update Import Paths — this is the most labor-intensive step), then Phase 5 (Update Documentation), Phase 6 (Update Tests), and finally Phase 7 (Cleanup). The `src/` layout means pip installs from `src/dapper/` while the repo root holds configuration, tests, docs, and dev-only scripts.
