"""
Global keybinding definitions for the Dataset Viewer TUI.

ALL screens and modals MUST use these bindings. Do NOT define bindings
inline in individual views — import from here instead.

Binding groups:
    GLOBAL_BINDINGS:     Always present (quit)
    BACK_BINDINGS:       Non-root screens (escape/b → go_back)
    VIM_NAV_BINDINGS:    Vim-style j/k/g/G navigation
    PANEL_BINDINGS:      Dual-pane panel switching (h/l/tab/arrows)
    TREE_BINDINGS:       Tree view actions (expand/collapse/detail)
    PAGE_BINDINGS:       Paginated list navigation (n/p/g/G pages)
    MODAL_BINDINGS:      Modal close/dismiss

Composite groups:
    SINGLE_PANE_BINDINGS:  GLOBAL + BACK + VIM_NAV
    DUAL_PANE_BINDINGS:    GLOBAL + BACK + VIM_NAV + PANEL
"""

from textual.binding import Binding


# ── Always present ──────────────────────────────────────────────────
GLOBAL_BINDINGS = [
    Binding("q", "quit", "Quit", show=False),
    Binding("m", "show_detail", "Detail", show=True),
    Binding("ctrl+t", "change_app_theme", "App Theme", show=True),
    Binding("ctrl+y", "change_syntax_theme", "Syntax", show=True),
]

# ── Non-root screens ────────────────────────────────────────────────
BACK_BINDINGS = [
    Binding("escape", "go_back", "Back", show=True),
    Binding("b", "go_back", "Back", show=False),
]

# ── Vim-style navigation (j/k/g/G) ─────────────────────────────────
VIM_NAV_BINDINGS = [
    Binding("j", "vim_down", "Down", show=False),
    Binding("k", "vim_up", "Up", show=False),
    Binding("g", "vim_top", "Top", show=False),
    Binding("G", "vim_bottom", "Bottom", show=False),
]

# ── Dual-pane panel switching ───────────────────────────────────────
PANEL_BINDINGS = [
    Binding("h", "vim_left", "Left Panel", show=False),
    Binding("l", "vim_right", "Right Panel", show=False),
    Binding("left", "vim_left", "Left Panel", show=False),
    Binding("right", "vim_right", "Right Panel", show=False),
    Binding("tab", "switch_panel", "Switch Panel", show=True),
]

# ── Tree view actions ──────────────────────────────────────────────
TREE_BINDINGS = [
    Binding("e", "expand_all", "Expand All"),
    Binding("c", "collapse_all", "Collapse All"),
]

# ── Paginated list navigation ──────────────────────────────────────
PAGE_BINDINGS = [
    Binding("n", "next_page", "Next Page", show=True),
    Binding("p", "prev_page", "Prev Page", show=True),
]

# ── Modal bindings ─────────────────────────────────────────────────
MODAL_BINDINGS = [
    Binding("escape", "close", "Close"),
    Binding("enter", "close", "Close"),
    Binding("q", "quit", "Quit App"),
    Binding("j", "scroll_down", "Down", show=True),
    Binding("k", "scroll_up", "Up", show=True),
    Binding("g", "scroll_home", "Top", show=True),
    Binding("G", "scroll_end", "Bottom", show=True),
    Binding("h", "vim_left", "Left", show=True),
    Binding("l", "vim_right", "Right", show=True),
    Binding("down", "scroll_down", "Down", show=False),
    Binding("up", "scroll_up", "Up", show=False),
    Binding("left", "vim_left", "Left", show=False),
    Binding("right", "vim_right", "Right", show=False),
]


# ── Composite groups ───────────────────────────────────────────────

# Single-pane screens: navigation + back + quit
SINGLE_PANE_BINDINGS = GLOBAL_BINDINGS + BACK_BINDINGS + VIM_NAV_BINDINGS

# Dual-pane screens: navigation + back + quit + panel switching
DUAL_PANE_BINDINGS = GLOBAL_BINDINGS + BACK_BINDINGS + VIM_NAV_BINDINGS + PANEL_BINDINGS
