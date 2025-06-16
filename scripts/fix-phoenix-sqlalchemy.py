#!/usr/bin/env python3
"""Fix Phoenix SQLAlchemy UnboundLocalError issue."""

import os
import sys


def fix_phoenix_models():
    """Fix the UnboundLocalError in Phoenix models.py."""

    # Find the models.py file
    models_path = "/Users/mikeedwards/.pyenv/versions/3.11.10/lib/python3.11/site-packages/phoenix/db/models.py"

    if not os.path.exists(models_path):
        print(f"❌ Phoenix models.py not found at {models_path}")
        return False

    # Read the file
    with open(models_path) as f:
        content = f.read()

    # Fix the problematic line 349
    # The issue is: assert isinstance(ans := value.model_dump(), str)
    # The walrus operator fails in some Python versions

    # Replace the problematic pattern
    old_code = """    def process_bind_param(
        self, value: Optional[TraceRetentionCronExpression], _: Dialect
    ) -> Optional[str]:
        assert isinstance(value, TraceRetentionCronExpression)
        assert isinstance(ans := value.model_dump(), str)
        return ans"""

    new_code = """    def process_bind_param(
        self, value: Optional[TraceRetentionCronExpression], _: Dialect
    ) -> Optional[str]:
        assert isinstance(value, TraceRetentionCronExpression)
        ans = value.model_dump()
        assert isinstance(ans, str)
        return ans"""

    if old_code in content:
        content = content.replace(old_code, new_code)
        print("✅ Fixed TraceRetentionCronExpression process_bind_param")
    else:
        print(
            "⚠️  TraceRetentionCronExpression pattern not found, trying alternative fix..."
        )

        # Try line-by-line replacement
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if "assert isinstance(ans := value.model_dump(), str)" in line:
                # Replace this line and add the next line
                lines[i] = "        ans = value.model_dump()"
                lines.insert(i + 1, "        assert isinstance(ans, str)")
                print(f"✅ Fixed line {i+1}")
                content = "\n".join(lines)
                break

    # Check for similar issues in _TraceRetentionRule
    old_rule_code = """    def process_bind_param(
        self, value: Optional[TraceRetentionRule], _: Dialect
    ) -> Optional[bytes]:
        assert isinstance(value, TraceRetentionRule)
        return msgpack.packb(ans := value.model_dump(mode="json"))"""

    new_rule_code = """    def process_bind_param(
        self, value: Optional[TraceRetentionRule], _: Dialect
    ) -> Optional[bytes]:
        assert isinstance(value, TraceRetentionRule)
        ans = value.model_dump(mode="json")
        return msgpack.packb(ans)"""

    if old_rule_code in content:
        content = content.replace(old_rule_code, new_rule_code)
        print("✅ Fixed TraceRetentionRule process_bind_param")
    else:
        # Try line-by-line for this one too
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if 'return msgpack.packb(ans := value.model_dump(mode="json"))' in line:
                lines[i] = '        ans = value.model_dump(mode="json")'
                lines.insert(i + 1, "        return msgpack.packb(ans)")
                print(f"✅ Fixed TraceRetentionRule at line {i+1}")
                content = "\n".join(lines)
                break

    # Write the fixed content back
    try:
        with open(models_path, "w") as f:
            f.write(content)
        print(f"✅ Successfully patched {models_path}")
        return True
    except PermissionError:
        print("❌ Permission denied. Creating a patched copy...")

        # Create a local patched version
        patched_path = "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/phoenix_models_patched.py"
        with open(patched_path, "w") as f:
            f.write(content)

        print(f"✅ Created patched copy at {patched_path}")
        print("To use it, you'll need to run Phoenix with a custom PYTHONPATH")
        return True


if __name__ == "__main__":
    print("🔧 Fixing Phoenix SQLAlchemy issues...")
    if fix_phoenix_models():
        print("\n✅ Phoenix SQLAlchemy issues fixed!")
        print("You can now start Phoenix server normally.")
    else:
        print("\n❌ Failed to fix Phoenix SQLAlchemy issues")
        sys.exit(1)
