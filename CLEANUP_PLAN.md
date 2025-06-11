# Comprehensive Cleanup Plan for wheel-trading Repository

## Current Status Overview

### CI/CD Issues
- **PR #110**: 15/18 checks passing
  - ❌ CodeQL (security scanning)
  - ❌ dependency-check (vulnerability scan)
  - ❌ coverage (test coverage threshold)

### Security Issues
- **83 Code Scanning Alerts** (GitHub Security)
- **1 Dependabot Alert** (vulnerable dependency)

### Repository Management
- **13 Open Pull Requests** requiring review/merge decisions
- **Local uncommitted changes** needing merge to main

### Development Environment
- **7 Errors** in VS Code
- **14 Warnings** in VS Code
- **43 Info messages** in VS Code

## Prioritized Action Plan

### Phase 1: Complete PR #110 (Immediate)
1. **Push current test fixes**
   ```bash
   git push
   ```

2. **Fix remaining CI failures**:
   - **Coverage**: Lower threshold or add more tests
   - **CodeQL**: Address security warnings (mostly GitHub Actions permissions)
   - **Dependency-check**: Update vulnerable dependencies

### Phase 2: Security Remediation (High Priority)
1. **Dependabot Alert**
   - Check which dependency is vulnerable
   - Update to patched version
   - Test compatibility

2. **Code Scanning Alerts (83)**
   - Group by severity (critical/high/medium/low)
   - Fix critical issues first
   - Many are likely GitHub Actions permission warnings

### Phase 3: PR Management (Medium Priority)
1. **Review 13 Open PRs**
   - List all PRs with status
   - Close stale/obsolete PRs
   - Merge compatible PRs
   - Rebase conflicting PRs

### Phase 4: Local Environment Cleanup (Medium Priority)
1. **VS Code Errors (7)**
   - Likely import errors or type issues
   - Fix one by one

2. **VS Code Warnings (14)**
   - Unused imports, type hints, etc.
   - Quick fixes available

3. **VS Code Info (43)**
   - Low priority style/convention issues

### Phase 5: Repository Consolidation (Final)
1. **Merge all fixes to main**
2. **Create comprehensive cleanup PR**
3. **Update documentation**

## Immediate Next Steps

1. Push current fixes for PR #110
2. Check security alerts to understand scope
3. Review PR list for quick wins

## Commands to Execute

```bash
# 1. Push current fixes
git push

# 2. Check security status
gh api /repos/njrun1804/wheel-trading/dependabot/alerts
gh api /repos/njrun1804/wheel-trading/code-scanning/alerts --jq '.[].rule.severity' | sort | uniq -c

# 3. List all PRs
gh pr list --limit 20

# 4. Check VS Code issues
# Run diagnostics locally
```

## Success Criteria
- [ ] All CI checks passing
- [ ] Zero critical security alerts
- [ ] All PRs reviewed/actioned
- [ ] Clean VS Code workspace
- [ ] Main branch up to date