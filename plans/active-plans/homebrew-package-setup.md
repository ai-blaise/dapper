# Plan: Homebrew Tap for Dapper

## Overview

Create a Homebrew tap (`blaise/homebrew-tap`) to enable `brew install blaise/tap/dapper`. The tap lives in a **separate GitHub repository** from the main dapper repo. This plan uses `uv pip install` with `uv` as a dependency for fast, reliable installs.

## Repository Structure

```
homebrew-tap/
├── README.md
└── Formula/
    └── dapper.rb
```

## Step-by-Step Plan

### Step 1: Create GitHub Repository

**What:** Create the `blaise/homebrew-tap` GitHub repository.
**Why:** Homebrew taps require a dedicated repo following the `homebrew-<name>` naming convention.
**Source:** [Homebrew Taps Documentation](https://docs.brew.sh/Taps)

### Step 2: Create Formula Directory

**What:** Create `Formula/` directory in the tap repo.
**Why:** Homebrew looks for formula files in this directory.
**Source:** Standard Homebrew tap structure.

### Step 3: Write Formula/dapper.rb

**What:** Create the Homebrew formula using `uv pip install` with `uv` as a dependency.
**Why:** `uv` is faster than pip and already the project's package manager. Using `uv pip install --target` installs the package into the Homebrew prefix without a virtualenv, keeping the formula simple.
**Source:** [Homebrew Formula Cookbook](https://docs.brew.sh/Formula-Cookbook), `uv pip install --target` documentation.

```ruby
class Dapper < Formula
  desc "Dataset Absurdly Powerful Parser of Epic Robustness"
  homepage "https://github.com/blaise/dapper"
  url "https://github.com/blaise/dapper/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "TBD"
  license "MIT"
  depends_on "uv"
  depends_on "python@3.14"

  def install
    system "uv", "pip", "install", "--target", prefix, "."
    bin.install_symlink "#{prefix}/bin/dapper" => "dapper"
  end

  def caveats
    <<~EOS
      The interactive TUI can be launched with:
        dapper view <dataset>

      Note: dapper view requires an interactive terminal.
    EOS
  end

  test do
    assert_match "list", shell_output("#{bin}/dapper --help")
    assert_match "show", shell_output("#{bin}/dapper --help")
    assert_match "search", shell_output("#{bin}/dapper --help")
    assert_match "stats", shell_output("#{bin}/dapper --help")
  end
end
```

### Step 4: Calculate SHA256 for Initial Release

**What:** Compute the SHA256 checksum of the dapper release tarball.
**Why:** Homebrew requires a SHA256 to verify the download integrity.
**Source:** Standard Homebrew requirement for formula `url`.

```bash
curl -sL https://github.com/blaise/dapper/archive/refs/tags/v0.1.0.tar.gz | shasum -a 256
```

Replace `"TBD"` in the formula with the computed hash.

### Step 5: Write README.md for the Tap

**What:** Create the tap's README with installation instructions and update workflow.
**Why:** Serves as documentation for users discovering the tap and maintainers updating the formula.
**Source:** Homebrew README conventions.

```markdown
# Homebrew Tap for Dapper

A custom Homebrew tap for [dapper](https://github.com/blaise/dapper) — Dataset Absurdly Powerful Parser of Epic Robustness.

## Installation

```bash
brew tap blaise/tap
brew install blaise/tap/dapper
```

## Usage

```bash
dapper list dataset.jsonl
dapper show dataset.jsonl 0
dapper search dataset.jsonl "query"
dapper stats dataset.jsonl
dapper parse dataset.jsonl -o output.jsonl
dapper split dataset.jsonl -n 4
dapper mix datasets/ -o output.parquet
dapper view dataset.jsonl
```

## Updating

When a new release is tagged on [blaise/dapper](https://github.com/blaise/dapper):

```bash
# Clone the tap repo
git clone https://github.com/blaise/homebrew-tap
cd homebrew-tap

# Calculate new SHA256
curl -sL https://github.com/blaise/dapper/archive/refs/tags/vX.Y.Z.tar.gz | shasum -a 256

# Update Formula/dapper.rb with the new version and SHA256
$EDITOR Formula/dapper.rb

git commit -m "dapper vX.Y.Z"
git push
```

## Uninstalling

```bash
brew uninstall blaise/tap/dapper
brew untap blaise/tap
```

## Development

```bash
# Test formula locally before pushing
brew install --build-from-source ./Formula/dapper.rb
brew test blaise/tap/dapper

# Audit the formula
brew audit --strict Formula/dapper.rb
```

## Repository Structure

```
homebrew-tap/
├── README.md
└── Formula/
    └── dapper.rb
```
```

### Step 6: Push to GitHub

**What:** Commit and push the initial tap structure to `blaise/homebrew-tap`.
**Why:** Makes the tap publicly available for `brew tap blaise/tap`.
**Source:** Homebrew auto-discovers taps from GitHub `homebrew-<repo>` repos.

## Files to Create

| File | Purpose |
|------|---------|
| `Formula/dapper.rb` | Homebrew formula (Ruby) |
| `README.md` | Tap documentation |

## Verification

```bash
# 1. Tap the repository
brew tap blaise/tap

# 2. Install dapper
brew install blaise/tap/dapper

# 3. Verify CLI works
dapper --help

# 4. Test a command
echo '{"messages":[{"role":"user","content":"test"}]}' | dapper list /dev/stdin
```

## Release Update Workflow

When `blaise/dapper` releases a new version `vX.Y.Z`:

1. Compute SHA256 of `https://github.com/blaise/dapper/archive/refs/tags/vX.Y.Z.tar.gz`
2. Update `Formula/dapper.rb`:
   - Change `url` to point to new tag
   - Update `sha256` with computed hash
3. Commit and push to `blaise/homebrew-tap`
4. Users get the update on next `brew upgrade`

## Risks / Clarifications

### 1. Python Version

@architect: The refactoring plan (Plan 1) uses `requires-python = ">=3.12"` in pyproject.toml. Homebrew's current Python is `python@3.14`. Which should the formula use?

| Option | Description |
|--------|-------------|
| `python@3.14` | Homebrew default, current stable |
| `python@3.12` | Matches existing pyproject.toml constraint |

**Recommendation:** `python@3.14` unless there is a specific reason to pin to 3.12.

### 2. uv Dependency

@architect: The formula uses `depends_on "uv"`. This means `uv` is installed automatically as part of the dapper installation. Confirm this is acceptable — `uv` is lightweight (~5MB) and one-time install.

### 3. Formula Approach

@architect: The formula uses `uv pip install --target`. An alternative was `Language::Python::Virtualenv` with explicit resource blocks for each dependency (more "Homebrew-native" but verbose). The chosen approach is simpler and matches the team's existing `uv` workflow. Confirm this is acceptable.

## Execution

@minimax-m2: Once the architect approves, execute the steps in order. Create the GitHub repo, write the formula and README, then verify with `brew install --build-from-source`. The formula uses `uv pip install --target` to install directly into the Homebrew prefix, then symlinks the `dapper` binary. This approach was chosen over `Language::Python::Virtualenv` because it is simpler and matches the team's existing `uv` workflow.
