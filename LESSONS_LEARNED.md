# Lessons Learned from MCP Server Setup

## What Went Wrong

1. **Feature Creep**: Started with trading bot needs, ended with 19 MCP servers including observability tools to debug the MCP servers themselves

2. **Script Proliferation**: Created 50+ shell scripts, each trying to fix issues created by previous scripts

3. **Wrong Assumptions**:
   - Assumed all `@modelcontextprotocol/server-*` packages existed on npm (they don't)
   - Assumed more monitoring would help debug issues (it added complexity)
   - Assumed performance optimization was needed before basic functionality

4. **Debugging Anti-Patterns**:
   - Created wrappers instead of fixing root causes
   - Suppressed errors instead of addressing them
   - Added complexity to solve complexity

## Key Insights

### Start Simple
- 3 working servers > 19 broken servers
- Validate each component before adding more
- Complex systems fail in complex ways

### Single Source of Truth
- One configuration file
- One launcher script
- One diagnostic tool

### Clear Failure Modes
- Servers should fail loudly with actionable errors
- Not: "BrokenPipeError is normal behavior"
- Yes: "Error: ripgrep binary not found. Install with: brew install ripgrep"

### Avoid Premature Optimization
- No CPU affinity before servers work
- No caching before understanding load
- No observability before basic functionality

## Better Approach

```bash
# Minimal viable setup
mcp-servers-minimal.json  # 3 essential servers
start-claude.sh          # Simple launcher
mcp-doctor.py           # Health check

# Gradual expansion (if needed)
1. Validate core functionality
2. Add one server at a time
3. Document why each server is needed
4. Test in isolation before integration
```

## Architecture Principles

1. **Independence**: Each server works standalone
2. **Testability**: Can validate without running Claude
3. **Debuggability**: Clear errors, simple logs
4. **Maintainability**: Obvious what each piece does

## Red Flags to Avoid

- Creating scripts to fix scripts
- Adding monitoring to debug monitoring  
- "Temporary" workarounds that become permanent
- Silent error suppression
- Multiple versions of the same functionality

## The Real Lesson

**Distributed systems are hard enough without making them harder.**

When debugging:
1. Simplify first
2. Understand the actual problem
3. Fix root causes, not symptoms
4. Validate fixes work in isolation
5. Only then integrate

The goal isn't to have the most servers or the most sophisticated setup. The goal is to have a working system that helps accomplish the actual task (in this case, trading bot development).