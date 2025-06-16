#!/usr/bin/env python3
"""
Bolt Uninstallation Script
Cleanly removes the 8-agent hardware-accelerated system

Features:
- Removes system and user symlinks
- Cleans up shell configuration files
- Optional removal of installed packages
- Backup of configuration before removal
- Comprehensive cleanup validation
"""

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def get_bolt_paths() -> list[Path]:
    """Find all Bolt-related paths in the system"""
    paths = []

    # System symlinks
    system_locations = [Path("/usr/local/bin/bolt"), Path("/usr/bin/bolt")]

    # User symlinks
    user_locations = [
        Path.home() / ".local" / "bin" / "bolt",
        Path.home() / "bin" / "bolt",
    ]

    # Check all locations
    for path in system_locations + user_locations:
        if path.exists():
            paths.append(path)

    return paths


def backup_shell_configs() -> Path:
    """Backup shell configuration files before modification"""
    backup_dir = (
        Path.home()
        / ".bolt_uninstall_backup"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    backup_dir.mkdir(parents=True, exist_ok=True)

    shell_files = [
        Path.home() / ".zshrc",
        Path.home() / ".bashrc",
        Path.home() / ".bash_profile",
        Path.home() / ".profile",
    ]

    backed_up = []
    for shell_file in shell_files:
        if shell_file.exists():
            backup_file = backup_dir / shell_file.name
            shutil.copy2(shell_file, backup_file)
            backed_up.append(shell_file.name)

    if backed_up:
        print(f"✅ Backed up shell configs: {', '.join(backed_up)}")
        print(f"   Backup location: {backup_dir}")

    return backup_dir


def clean_shell_configs() -> bool:
    """Remove Bolt-related lines from shell configuration files"""
    shell_files = [
        Path.home() / ".zshrc",
        Path.home() / ".bashrc",
        Path.home() / ".bash_profile",
        Path.home() / ".profile",
    ]

    cleaned_files = []
    bolt_markers = [
        "# Added by Bolt installer",
        "# Bolt installation",
        "bolt_executable",
        "/bolt:",
        ":bolt/",
    ]

    for shell_file in shell_files:
        if not shell_file.exists():
            continue

        try:
            lines = shell_file.read_text().splitlines()
            original_count = len(lines)

            # Filter out Bolt-related lines
            cleaned_lines = []
            skip_next = False

            for line in lines:
                if skip_next:
                    skip_next = False
                    continue

                # Check if line contains Bolt markers
                is_bolt_line = any(marker in line for marker in bolt_markers)

                if is_bolt_line:
                    # If this is a comment line, skip the next line too (likely the actual command)
                    if line.strip().startswith("#") and "Bolt" in line:
                        skip_next = True
                    continue

                cleaned_lines.append(line)

            # Write back if changes were made
            if len(cleaned_lines) != original_count:
                shell_file.write_text("\n".join(cleaned_lines) + "\n")
                cleaned_files.append(shell_file.name)
                print(
                    f"   ✅ Cleaned {shell_file.name} ({original_count - len(cleaned_lines)} lines removed)"
                )

        except Exception as e:
            print(f"   ⚠️ Could not clean {shell_file.name}: {e}")

    if cleaned_files:
        print(f"✅ Cleaned shell configurations: {', '.join(cleaned_files)}")
        return True
    else:
        print("ℹ️ No shell configurations needed cleaning")
        return True


def remove_symlinks(paths: list[Path]) -> bool:
    """Remove Bolt symlinks"""
    if not paths:
        print("ℹ️ No Bolt symlinks found")
        return True

    removed = []
    failed = []

    for path in paths:
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
                removed.append(str(path))
            elif path.is_dir():
                shutil.rmtree(path)
                removed.append(str(path))

        except PermissionError:
            print(f"   ⚠️ Permission denied: {path}")
            failed.append(str(path))
        except Exception as e:
            print(f"   ⚠️ Could not remove {path}: {e}")
            failed.append(str(path))

    if removed:
        print(f"✅ Removed symlinks: {', '.join(removed)}")

    if failed:
        print(f"❌ Failed to remove: {', '.join(failed)}")
        print("   You may need to run with sudo or remove manually")
        return False

    return True


def uninstall_packages(packages: list[str]) -> bool:
    """Optionally uninstall Bolt-specific packages"""
    print(f"\n🗑️ Optional: Remove {len(packages)} Bolt-specific packages?")
    print("   This will remove packages that may be used by other applications")
    print(
        "   Packages to remove:",
        ", ".join(packages[:5]) + ("..." if len(packages) > 5 else ""),
    )

    response = input("   Remove packages? [y/N]: ").strip().lower()

    if response != "y":
        print("   Skipping package removal")
        return True

    print("📦 Removing packages...")
    failed_packages = []

    for package in packages:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", package],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                print(f"   ✅ Removed {package}")
            else:
                print(f"   ⚠️ Could not remove {package} (may not be installed)")

        except subprocess.TimeoutExpired:
            print(f"   ⏰ Timeout removing {package}")
            failed_packages.append(package)
        except Exception as e:
            print(f"   ❌ Error removing {package}: {e}")
            failed_packages.append(package)

    if failed_packages:
        print(f"⚠️ Failed to remove: {', '.join(failed_packages)}")
        return False

    return True


def validate_uninstallation() -> bool:
    """Verify that Bolt has been completely removed"""
    print("\n🔍 Validating uninstallation...")

    # Check for remaining symlinks
    remaining_paths = get_bolt_paths()
    if remaining_paths:
        print(f"   ⚠️ Found remaining paths: {remaining_paths}")
        return False
    else:
        print("   ✅ No remaining symlinks found")

    # Test that bolt command is no longer available
    try:
        result = subprocess.run(
            ["bolt", "--help"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print("   ⚠️ Bolt command still accessible")
            return False
        else:
            print("   ✅ Bolt command no longer accessible")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("   ✅ Bolt command no longer accessible")

    # Check for configuration remnants
    shell_files = [
        Path.home() / ".zshrc",
        Path.home() / ".bashrc",
        Path.home() / ".bash_profile",
    ]

    remaining_configs = []
    for shell_file in shell_files:
        if shell_file.exists():
            content = shell_file.read_text()
            if "bolt" in content.lower() and (
                "export path" in content.lower() or "alias" in content.lower()
            ):
                remaining_configs.append(shell_file.name)

    if remaining_configs:
        print(f"   ⚠️ Found Bolt references in: {', '.join(remaining_configs)}")
        return False
    else:
        print("   ✅ No configuration remnants found")

    return True


def create_uninstall_log(backup_dir: Path, success: bool) -> None:
    """Create uninstallation log"""
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "backup_location": str(backup_dir),
        "python_executable": sys.executable,
        "platform": sys.platform,
    }

    log_file = backup_dir / "uninstall.log"
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"📝 Uninstall log saved to: {log_file}")


def main():
    """Main uninstallation process"""
    start_time = time.time()

    print("🗑️ Uninstalling Bolt - 8-Agent Hardware-Accelerated System")
    print("=" * 60)

    # Confirmation
    print("This will remove:")
    print("• Bolt executable and symlinks")
    print("• Shell configuration modifications")
    print("• Optionally: installed Python packages")
    print()

    response = input("Continue with uninstallation? [y/N]: ").strip().lower()
    if response != "y":
        print("Uninstallation cancelled")
        sys.exit(0)

    # Find Bolt paths
    bolt_paths = get_bolt_paths()
    print(f"\n🔍 Found {len(bolt_paths)} Bolt installations:")
    for path in bolt_paths:
        print(f"   - {path}")

    # Backup shell configurations
    print("\n💾 Creating backup...")
    backup_dir = backup_shell_configs()

    # Clean shell configurations
    print("\n🧹 Cleaning shell configurations...")
    shell_success = clean_shell_configs()

    # Remove symlinks
    print("\n🗑️ Removing Bolt symlinks...")
    symlink_success = remove_symlinks(bolt_paths)

    # Optional package removal
    bolt_packages = ["mlx", "mlx-lm", "pynvml", "orjson", "lmdb", "uvloop"]
    package_success = uninstall_packages(bolt_packages)

    # Validate uninstallation
    validation_success = validate_uninstallation()

    # Overall success
    overall_success = shell_success and symlink_success and validation_success
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    if overall_success:
        print(f"✅ Bolt uninstallation completed successfully in {elapsed:.1f}s!")
    else:
        print(f"⚠️ Bolt uninstallation completed with issues in {elapsed:.1f}s")

    print("\n📋 Summary:")
    print(f"   Shell configs: {'✅' if shell_success else '❌'}")
    print(f"   Symlinks: {'✅' if symlink_success else '❌'}")
    print(f"   Packages: {'✅' if package_success else '❌'}")
    print(f"   Validation: {'✅' if validation_success else '❌'}")

    print(f"\n💾 Backup location: {backup_dir}")
    print("   (You can restore from here if needed)")

    if not overall_success:
        print("\n⚠️ Some manual cleanup may be required:")
        print("   1. Check remaining symlinks manually")
        print("   2. Review shell configuration files")
        print("   3. Restart your terminal")

    create_uninstall_log(backup_dir, overall_success)

    print(
        "\n👋 Bolt has been uninstalled. Restart your terminal to complete the process."
    )


if __name__ == "__main__":
    main()
