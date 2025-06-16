# EMERGENCY: File Descriptor Exhaustion Recovery

## IMMEDIATE ACTIONS AFTER TERMINAL RESTART

1. **Check System State:**
```bash
ulimit -n                    # Check current FD limit
lsof | wc -l                # Count open files
ps aux | grep python        # Find Python processes
ps aux | grep meta          # Find meta processes
```

2. **Kill Hanging Processes:**
```bash
pkill -f "meta_daemon"
pkill -f "meta_system" 
pkill -f "meta_monitoring"
kill -9 39124 2>/dev/null || true
kill -9 84182 2>/dev/null || true
```

3. **Clean Database Files:**
```bash
cd /Users/mikeedwards/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading
find . -name "*.db-wal" -delete
find . -name "*.db-shm" -delete
find . -name "*.pid" -delete
```

4. **Increase FD Limits:**
```bash
ulimit -n 4096              # Temporary increase
echo 'ulimit -n 4096' >> ~/.zshrc  # Permanent increase
```

5. **Verify Recovery:**
```bash
echo "Test command works"   # Should work now
ls -la                      # Should work now
ulimit -n                   # Should show 4096
```

## CONFIG CHANGES TO PREVENT RECURRENCE

### ~/.zshrc additions:
```bash
# Resource limits
ulimit -n 4096
ulimit -u 2048

# Monitoring aliases
alias check-fds='lsof -p $$ | wc -l'
alias fd-usage='echo "FDs: $(lsof -p $$ | wc -l)/$(ulimit -n)"'
alias resource-check='echo "FDs: $(lsof | wc -l) Memory: $(ps -eo rss | awk "{sum+=\$1} END {print sum/1024\"MB\"}")"'

# Auto cleanup on shell exit
trap 'pkill -f meta_daemon; pkill -f meta_system' EXIT
```

### System-wide limits (/etc/launchd.conf):
```
limit maxfiles 8192 8192
limit maxproc 2048 2048
```

## ROOT CAUSES FIXED
- SQLite connections without context managers
- Background processes without cleanup handlers  
- Missing atexit handlers in daemon processes
- No resource monitoring or limits

## PREVENTION IMPLEMENTED
- Resource management system in src/unity_wheel/utils/
- Automatic cleanup on exit
- File descriptor monitoring
- Process lifecycle management