# Borrowing Cost Analysis Guide

## Overview

The borrowing cost analyzer provides pure mathematical analysis for capital allocation decisions in a tax-free environment. It helps determine whether to use borrowed funds for Unity wheel positions or pay down existing debt.

## Key Features

### 1. Pure Mathematics - No Safety Factors
- **Original**: 1.5x safety multiplier, 25% tax adjustment
- **Now**: Pure borrowing rates with no adjustments
- **Tax-free**: All returns are gross returns

### 2. Exact Interest Calculations
- Daily compounding for accurate cost calculations
- Effective Annual Rate (EAR) vs stated APR
- Present value and NPV calculations

### 3. Your Debt Situation
```
Amex Personal Loan: $45,000 @ 7.00% APR (7.25% EAR)
Schwab Margin: Available @ 10.00% APR (10.52% EAR)
Daily Cost on Amex: $8.63
```

## Pure Math Decision Rules

### Simple Hurdle Rates
- **Amex**: Need >7% annualized return to beat paying it down
- **Schwab**: Need >10% annualized return to justify borrowing
- **Unity Wheel**: Typically returns 15-25% annualized

### Break-Even Analysis
For a 45-day Unity position borrowing 100% from Schwab:
- Borrowing cost: $434 on $35,000
- Break-even return: 1.24% total (10.06% annualized)
- Each 1% return = $350 profit
- Each 1% rate increase = $43 additional cost

## Usage Examples

### Quick Check
```python
from src.unity_wheel.risk import analyze_borrowing_decision

result = analyze_borrowing_decision(
    position_size=35000,
    expected_return=0.20,  # 20% annualized
    confidence=0.8,
    available_cash=10000
)

print(f"Action: {result.action}")  # 'invest' or 'paydown_debt'
print(f"Hurdle rate: {result.hurdle_rate:.1%}")
```

### Real-Time Rate Updates
```python
from src.unity_wheel.risk import BorrowingCostAnalyzer

def fetch_live_rate(source):
    return {"schwab_margin": 0.095}.get(source)

analyzer = BorrowingCostAnalyzer(rate_fetcher=fetch_live_rate, auto_update=True)
analyzer.update_rates()
```

### Advanced Analysis with NPV/IRR
```python
from src.unity_wheel.risk.pure_borrowing_analyzer import PureBorrowingAnalyzer

analyzer = PureBorrowingAnalyzer()
analysis = analyzer.analyze_investment(
    investment_amount=35000,
    expected_return=0.03,  # 3% total return over period
    holding_days=45,
    available_cash=0,
    loan_source="schwab"
)

print(f"NPV: ${analysis.npv:,.2f}")
print(f"IRR: {analysis.irr:.1%}" if analysis.irr else "IRR: N/A")
print(f"Break-even return: {analysis.break_even_return:.3%}")
print(f"Days to break even: {analysis.days_to_break_even}")
```

### Sensitivity Analysis
The analyzer automatically performs sensitivity analysis:
- 10% worse return impact on NPV
- 20% longer holding period impact
- 1% higher borrowing rate impact

## Key Insights

### Mathematical Facts
1. **Compound Interest**: Schwab at 10% APR = 10.52% EAR with daily compounding
2. **Opportunity Cost**: Not paying down 7% debt costs $8.63/day
3. **Time Value**: Shorter holding periods reduce borrowing cost proportionally
4. **No Taxes**: In tax-free environment, 20% gross = 20% net return

### Decision Framework
1. Calculate expected total return (not annualized)
2. Compare to borrowing cost for same period
3. If return > cost, mathematically profitable
4. Consider confidence level (reduces expected return)

### Unity Wheel Specifics
- Typical premium: 2-5% per 45-day cycle
- Annualized: 15-40% depending on volatility
- Most positions beat 7% Amex hurdle
- High volatility periods often beat 10% Schwab hurdle

## Configuration

### No Configuration Needed
The analyzer uses pure mathematics with no adjustable parameters:
- Interest rates from actual loan terms
- Daily compounding (industry standard)
- Tax-free environment (no adjustments)

### Customizing Loan Terms
```python
from src.unity_wheel.risk.borrowing_cost_analyzer import BorrowingSource

analyzer = BorrowingCostAnalyzer()
analyzer.add_source(BorrowingSource(
    name="heloc",
    balance=0,
    annual_rate=0.085,  # 8.5% APR
    is_revolving=True
))
```

## Examples in `/examples/core/`

1. **`pure_math_borrowing_demo.py`** - Complete mathematical analysis with NPV/IRR
2. **`borrowing_analysis_demo.py`** - Scenario comparison tool
3. **`quick_borrowing_check.py`** - Simple go/no-go decisions
4. **`integrated_decision_example.py`** - Full wheel + borrowing analysis

## Bottom Line

With pure mathematics and no safety factors:
- Paying down 7% debt = guaranteed 7% return
- Unity wheel typically beats both debt hurdles
- Tax-free environment improves all returns
- Use confidence level to adjust expected returns
