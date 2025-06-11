# PR Triage Recommendations

## Summary
13 open PRs, all created by njrun1804 (likely automated suggestions from AI tools)

## Recommended Actions

### 🟢 MERGE (Aligns with Simplification)
1. **#92 - Remove orphan files**
   - Cleans up unused code
   - Reduces complexity

2. **#84 - Remove deprecated Schwab ingestion module**
   - Removes unused code
   - You only need FRED & Databento

3. **#104 - Switch CI to Python 3.11**
   - Already done in our work today
   - Can close as completed

### 🟡 REVIEW CAREFULLY
4. **#110 - Implement dynamic risk-free rate in backtester** (Current PR)
   - This is the one we've been working on
   - Has all our fixes

5. **#95 - Update docs for src.unity_wheel imports**
   - Documentation updates are good
   - Check if still relevant

6. **#99 - Standardize shell script safety**
   - Good for consistency
   - But check if adds complexity

### 🔴 PROBABLY SKIP (Over-Engineering)
7. **#113 - Add portfolio permutation optimizer**
   - Complex optimization for a recommendation bot
   - YAGNI (You Aren't Gonna Need It)

8. **#112 - Implement walk-forward backtesting utility**
   - Advanced backtesting feature
   - Overkill for personal use

9. **#101 - Add size-based eviction for DuckDB cache**
   - Premature optimization
   - No evidence of cache issues

10. **#70 - Enhance risk checks and reporting**
    - More complexity
    - Current risk checks are sufficient

11. **#65 - Improve async data loading**
    - Performance optimization
    - Not needed for single-user bot

### 🤔 NEEDS CONTEXT
12. **#111 - Check minimum price history length**
    - Might be useful validation
    - Quick review needed

13. **#97 - Update container setup script**
    - Do you even use containers?
    - If not, close it

## Recommendation

1. **Close #104** - Already done
2. **Merge #92, #84** - Remove unused code
3. **Finish #110** - Current work with all fixes
4. **Review #95, #99, #111** - Quick checks
5. **Close the rest** - Over-engineering for private bot

This aligns with today's theme: **simplify for a private recommendation bot**!
