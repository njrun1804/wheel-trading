# Claude Code Performance Quick Reference

## 🚀 Immediate Actions

```bash
source ~/.zshrc          # Activate shell optimizations
```

## 📊 Monitor Performance

```bash
perfmon                  # Beautiful system monitor (btop)
ulimit -n               # Check file descriptor limit (should be 65536)
echo $NODE_OPTIONS      # Verify Node.js memory (should be 8GB)
dust -n 10              # Check disk usage by largest dirs
```

## 🧹 Regular Maintenance

```bash
cleanup                  # Clean all caches (weekly)
./scripts/maintenance.sh # Full maintenance (weekly)
make clean              # Project-specific cleanup
```

## 💡 Best Practices for Speed

### When Claude Code is slow:

1. Close unused files in VS Code
2. Run `make clean` to remove build artifacts
3. Check disk space: `df -h`
4. Use specific file searches, not broad ones
5. Restart VS Code if needed

### Project organization:

- Keep large files (>10MB) out of the repository
- Use .gitignore for generated files
- Organize code in clear directory structures
- Name files descriptively

### Development tips:

```bash
nosleep make test       # Prevent sleep during long operations
git gc --aggressive     # Optimize git repository (monthly)
```

## 🛠️ Your Optimized Environment

- **M4 Pro MacBook**: 12 cores, 24GB RAM ✅
- **File descriptors**: 65,536 (256x increase)
- **Node.js heap**: 8GB allocated
- **VS Code**: Optimized for large codebases
- **Git**: Caching and preloading enabled

## 🔧 Troubleshooting

If optimizations aren't working:

```bash
# Re-run optimizations
./scripts/optimize-shell.sh
./scripts/mac-optimize.sh
source ~/.zshrc

# Check VS Code settings
cat .vscode/settings.json | grep -E "watcher|memory"
```
