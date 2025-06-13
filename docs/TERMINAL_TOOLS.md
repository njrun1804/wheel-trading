# Terminal Tools Reference

## Installed Development Tools

### File Navigation & Search
- **`eza`** - Modern ls with icons
  - `ls` → `eza --icons`
  - `ll` → `eza -l --icons`
  - `tree` → `eza --tree --icons`
  
- **`fd`** - Fast find alternative
  - `fd pattern` - Find files matching pattern
  - `fd -e py` - Find Python files
  
- **`fzf`** - Fuzzy finder
  - `Ctrl+R` - Fuzzy search command history
  - `Ctrl+T` - Fuzzy find files
  - `Alt+C` - Fuzzy cd to directory

- **`zoxide`** - Smart cd that learns
  - `z wheel` - Jump to wheel-trading dir
  - `zi` - Interactive selection

### Code Viewing & Editing
- **`bat`** - Better cat with syntax highlighting
  - `cat file.py` → `bat file.py`
  
- **`delta`** - Beautiful git diffs
  - Automatically used for `git diff`
  - Side-by-side view enabled

### System Monitoring
- **`btop`** - Resource monitor (replaces htop)
  - `btop` or `Cmd+Shift+T` in WezTerm
  
- **`duf`** - Better disk usage
  - `duf` or `Cmd+Shift+D` in WezTerm
  
- **`dust`** - Directory disk usage analyzer
  - `dust` or `du` (aliased)

### Git Tools
- **`lazygit`** - Terminal UI for git
  - `lg` or `Cmd+Shift+G` in WezTerm
  
- **`gh`** - GitHub CLI
  - `gh pr create` - Create pull request
  - `gh pr list` - List PRs

### Python Development
- **`ptpython`** - Better Python REPL
  - `ptpython` or `Cmd+Shift+I`
  - Auto-completion, syntax highlighting
  
- **`ipython`** - Enhanced interactive Python
  - `ipy` (aliased)
  
- **`icecream`** - Better debugging
  - `from icecream import ic`
  - `ic(variable)` - Pretty print with context

- **`httpie`** - User-friendly HTTP client
  - `http GET api.example.com`

### Shell Enhancements
- **`starship`** - Fast, customizable prompt
  - Shows git status, Python env, time
  - Config at ~/.config/starship.toml

- **`tldr`** - Simplified man pages
  - `tldr git` - Quick git reference
  - `tldr pytest` - Testing examples

### Other Tools
- **`yq`** - YAML processor
  - `yq '.key' config.yaml` - Extract values
  - `yq -i '.key = "value"' file.yaml` - Edit in place

- **`ruff`** - Fast Python linter
- **`black`** - Code formatter
- **`mypy`** - Type checker

## WezTerm Shortcuts

### Navigation
- `Cmd+D` - Split pane horizontally
- `Cmd+Shift+D` - Split pane vertically
- `Cmd+Alt+H/L/J/K` - Navigate panes
- `Cmd+Alt+1/2/3` - Switch workspaces

### Wheel Trading
- `Cmd+Shift+R` - Run trading recommendation
- `Cmd+Shift+T` - Run tests
- `Cmd+Shift+O` - Launch orchestrator
- `Cmd+Shift+S` - Ultimate M4 Pro launch
- `Cmd+Shift+H` - Health check

### Development
- `Cmd+Shift+E` - Fuzzy find & edit Python files
- `Cmd+Shift+G` - Open lazygit
- `Cmd+Shift+I` - Interactive Python with imports
- `Cmd+Shift+P` - Performance profiling

### Quick Aliases
```bash
wheel      # cd to project
wrun       # run trading analysis
wtest      # run fast tests
worchestrate # launch orchestrator
wlog       # tail orchestrator log
whealth    # system health check
analyze    # trading analysis
```

## Tips
1. Use `z wheel` to quickly jump to project
2. `Ctrl+R` for fuzzy command history search
3. `bat` automatically pages long files
4. `lazygit` makes complex git operations simple
5. `ptpython` has better autocomplete than regular Python REPL