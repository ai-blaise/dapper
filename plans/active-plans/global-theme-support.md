# Global Theme Support for TUI

**Status:** PENDING

Add global theme support to the TUI, allowing users to configure and persist their preferred color theme.

## Requirements

1. **Config Location:** `config.json` in project root (not `.config`)
2. **CLI Flags:** `--app-theme` and `--syntax-theme` flags to override config at runtime
3. **Footer Display:** All relevant theme commands displayed in the footer
4. **Notifications:** Toast notification when theme changes
5. **Scope:** TUI only (not CLI)
6. **Default App Theme:** `textual-dark`
7. **Default Syntax Theme:** `monokai`

## Implementation Plan

### 1. Config Module (`utils/config.py`)

@architect: These should be in its own python file in utils dir.

@minimax-m2: Moving config module to `utils/config.py` - will create a new module alongside the existing `utils/data.py` and `utils/streaming.py`

Create a new module in the `utils/` directory with functions:
- `load_config() -> dict` - Load config from `config.json`
- `save_config(config: dict) -> None` - Save config to file
- `get_app_theme() -> str` - Get saved app theme or default ("textual-dark")
- `set_app_theme(theme_name: str) -> None` - Save app theme preference
- `get_syntax_theme() -> str` - Get saved syntax theme or default ("monokai")
- `set_syntax_theme(theme_name: str) -> None` - Save syntax theme preference

### 2. Config File (`config.json`)

**Committed to repo** - users can modify this file directly:
```json
{
  "theme": "atom-one-dark"
}
```

### 3. CLI Argument (`scripts/tui/app.py`)

Add `--app-theme` and `--syntax-theme` arguments to argparse:
```python
parser.add_argument("--app-theme", default=None, help="App theme name (e.g., nord, atom-one-dark)")
parser.add_argument("--syntax-theme", default=None, help="Syntax highlighting theme (e.g., monokai, dracula)")
```

### 4. Update `JsonComparisonApp` (`scripts/tui/app.py`)

**In `__init__`:**
- Store themes: 
  - `self._app_theme = app_theme_from_cli or load_app_theme() or "textual-dark"`
  - `self._syntax_theme = syntax_theme_from_cli or load_syntax_theme() or "monokai"`

**In `on_mount()`:**
- Set `self.theme = self._app_theme`
- Store `self._syntax_theme` for later use in modals

**New action methods:**
```python
def action_change_app_theme(self, theme_name: str) -> None:
    self.theme = theme_name
    set_app_theme(theme_name)
    self.notify(f"App theme changed to: {theme_name}")

def action_change_syntax_theme(self, theme_name: str) -> None:
    self._syntax_theme = theme_name
    set_syntax_theme(theme_name)
    self.notify(f"Syntax theme changed to: {theme_name}")
```

**Publish syntax theme for widgets:**
- Add `@signal` or pass to modals via message to enable live preview

### 5. Update Global Keybindings (`scripts/tui/keybindings.py`)

Add to `GLOBAL_BINDINGS`:
```python
Binding("ctrl+t", "change_theme", "Theme", show=True),
```

`show=True` ensures it appears in both footer and command palette.

### 6. Theme Priority

**App Theme:**
1. CLI argument (`--app-theme`) - highest priority
2. Config file (`config.json`)
3. Default (`textual-dark`) - lowest priority

**Syntax Theme:**
1. CLI argument (`--syntax-theme`) - highest priority
2. Config file (`config.json`)
3. Default (`monokai`) - lowest priority

### 7. Show Syntax Highlighting in Detail Modal

In `scripts/tui/widgets/field_detail_modal.py`:

- Replace `Static` with `Rich.Syntax` widget for JSON content:
  ```python
  from rich.syntax import Syntax
  # ...
  syntax = Syntax(
      formatted_value,
      "json",
      theme=self.app._syntax_theme,  # Uses Pygments theme from config
      line_numbers=False,
  )
  ```

- This applies Pygments-based syntax highlighting to JSON content using the stored `_syntax_theme`

## Available Themes

### Textual App Themes (registered)

All built-in Textual themes:

| Theme | Dark/Light | Notes |
|-------|------------|-------|
| textual-dark | Dark | Default |
| nord | Dark | |
| gruvbox | Dark | |
| tokyo-night | Dark | |
| atom-one-dark | Dark | |
| atom-one-light | Light | |
| solarized-light | Light | |
| solarized-dark | Dark | |

### Pygments Themes (for JSON syntax highlighting)

Separate field in config.json for syntax highlighting:

| Theme | Notes |
|-------|-------|
| monokai | Default for JSON |
| dracula | |
| nord | |
| gruvbox-dark | |
| solarized-dark | |
| solarized-light | |

## Config File (`config.json`)

Two separate theme fields:
```json
{
  "app_theme": "textual-dark",
  "syntax_theme": "monokai"
}
```

## Testing

After implementation, verify:
1. App starts with default theme when no config exists
2. `--theme nord` flag changes theme
3. Ctrl+T opens theme selection
4. Theme persists after app restart
5. Toast notification appears on theme change
6. Theme name appears in the detail modal footer