# Claude Code Performance Optimizations

## Quick Start

Run this command to apply all optimizations:

```bash
./scripts/optimize-shell.sh && source ~/.zshrc
```

## Applied Optimizations

### 1. VS Code Settings (✅ Applied)

- **File watcher exclusions**: Ignores node_modules, venv, coverage files
- **Search exclusions**: Speeds up file searches
- **Large file optimizations**: Better handling of large codebases
- **Telemetry disabled**: Reduces background processes

### 2. Shell Optimizations

Run `./scripts/optimize-shell.sh` to apply:

- **File descriptor limit**: Increased to 10,240
- **History size**: Increased to 50,000 entries
- **Git prompt**: Disabled slow features
- **Node.js memory**: Increased to 8GB
- **Python**: Disabled bytecode generation

### 3. System-Level Optimizations

- **Spotlight exclusions**: node_modules and venv folders
- **SSD usage**: Keep projects on SSD, not network drives

### 4. Claude Code Best Practices

#### Project Structure

- Keep files organized in clear directories
- Use descriptive file names
- Maintain a clean git repository

#### Context Management

- Close files you're not actively working on
- Use focused searches instead of broad ones
- Keep .gitignore updated with large/generated files

#### Command Usage

- Use batch operations when possible (multiple tool calls)
- Prefer Glob/Grep over Agent for specific searches
- Use TodoWrite to track complex tasks

#### Performance Tips

- Avoid working with files > 10MB
- Exclude build artifacts from version control
- Keep node_modules and venv in .gitignore
- Use `make clean` regularly to remove generated files

## Verification

Check if optimizations are working:

```bash
# Check file descriptor limit
ulimit -n

# Check Node.js memory
echo $NODE_OPTIONS

# Check VS Code settings
cat .vscode/settings.json | grep -E "watcherExclude|maxMemory"
```

## Troubleshooting

If Claude Code seems slow:

1. Check available disk space: `df -h`
2. Check system resources: `top` or Activity Monitor
3. Clear VS Code cache: `rm -rf ~/Library/Application\ Support/Code/Cache`
4. Restart VS Code
5. Run `make clean` to remove build artifacts
