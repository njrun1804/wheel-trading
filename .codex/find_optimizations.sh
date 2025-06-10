#!/bin/bash
# Find optimization opportunities for Codex

echo "🔍 FINDING OPTIMIZATION OPPORTUNITIES"
echo "====================================="

echo ""
echo "1. 🚨 EXCEPTION HANDLING"
echo "   Bare exceptions that need specific handling:"
rg "except\s*:" src/unity_wheel/ data_pipeline/ --type py -n | head -10
rg "except\s+Exception\s*:" src/unity_wheel/ data_pipeline/ --type py -n | head -10

echo ""
echo "2. 🚀 PERFORMANCE OPPORTUNITIES"
echo "   Loops that could be vectorized:"
rg "for.*in.*range" src/unity_wheel/ --type py -n | grep -v test | head -10
rg "for.*strike.*in" src/unity_wheel/ --type py -n | head -5

echo ""
echo "3. 📊 MISSING CONFIDENCE SCORES"
echo "   Functions that should return confidence:"
rg "def (calculate_|black_scholes)" src/unity_wheel/ --type py -A 3 | grep -B 3 -A 3 "return " | grep -v confidence | head -10

echo ""
echo "4. 🔧 HARDCODED VALUES"
echo "   Values that should be in config:"
rg "(position_size|num_contracts|contract_count)\s*=\s*[0-9]+" src/unity_wheel/ --type py -n | head -5
rg "volatility.*[<>].*[0-9]" src/unity_wheel/ --type py -n | head -5

echo ""
echo "5. 📈 MEMORY OPTIMIZATIONS"
echo "   Large object creation in loops:"
rg "for.*in.*:" src/unity_wheel/ --type py -A 5 | grep -B 1 -A 4 "np\.array\|list\(\|dict\(" | head -10

echo ""
echo "6. ⚡ FUNCTION COMPLEXITY"
echo "   Long functions that could be split:"
echo "   Functions with >50 lines:"
for file in $(find src/unity_wheel/ -name "*.py" -type f); do
    awk '/^def |^class |^async def / {
        if (func != "") print file ":" lineno ":" func " (" (NR-lineno) " lines)"
        func = $0; lineno = NR
    }
    END {
        if (func != "") print file ":" lineno ":" func " (" (NR-lineno+1) " lines)"
    }' file="$file" "$file" | awk -F'[:(]' '$NF+0 > 50'
done | head -10

echo ""
echo "🎯 PRIORITY RECOMMENDATIONS:"
echo "1. Replace bare 'except:' with specific exceptions"
echo "2. Vectorize loops using numpy operations"
echo "3. Add confidence scores to calculation functions"
echo "4. Move hardcoded values to config.yaml"
echo "5. Split complex functions into smaller pieces"

echo ""
echo "💡 Quick validation after changes:"
echo "   ./scripts/housekeeping.sh --unity-check"
echo "   pytest tests/test_wheel.py -v"
