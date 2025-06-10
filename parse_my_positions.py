#!/usr/bin/env python3
"""Parse your specific position data and create a manual input file."""

# Based on your data:
# Account Value: $212,001.67
# Cash: $41,001.67
# Buying Power: $293,546.00 (Cash + Borrowing)
# Positions:
# - U: 7,500 shares @ $24.84
# - U 07/18/2025 25.00 C: -75 calls @ $2.08

# Create manual input format
manual_input = """Account Value: $212,001.67
Cash: $41,001.67
Buying Power: $293,546.00

Positions:
U 7500 shares @ $24.84
U250718C00025000 -75 calls @ $2.08"""

# Save to file
with open("my_positions.txt", "w") as f:
    f.write(manual_input)

print("✅ Created my_positions.txt with your account data")
print("\n📋 Your positions:")
print("   U: 7,500 shares @ $24.84")
print("   U Jul 18 2025 $25 Call: -75 (short)")
print("\n🎯 Unity Exposure:")
print("   Stock Value: $186,600")
print("   Short Call Value: -$15,600")
print("   Net Position Value: $171,000")
print("\n💡 To get a recommendation:")
print("   python run.py --manual < my_positions.txt")
print("\n   Or interactively:")
print("   python run.py --manual")
print("   (then paste the contents of my_positions.txt)")
