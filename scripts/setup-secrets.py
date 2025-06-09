#!/usr/bin/env python3
"""Setup script for configuring secrets in Unity Wheel Trading Bot."""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unity_wheel.secrets import SecretManager, SecretProvider


def check_gcp_setup() -> bool:
    """Check if GCP is properly configured."""
    try:
        # Check for gcloud CLI
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            return False
        
        project_id = result.stdout.strip()
        if not project_id or project_id == "(unset)":
            return False
        
        print(f"✓ GCP Project configured: {project_id}")
        
        # Check for application default credentials
        result = subprocess.run(
            ["gcloud", "auth", "application-default", "print-access-token"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            print("✓ GCP Application Default Credentials configured")
            return True
        else:
            print("✗ GCP Application Default Credentials not configured")
            return False
            
    except FileNotFoundError:
        print("✗ gcloud CLI not found")
        return False


def setup_gcp() -> None:
    """Guide user through GCP setup."""
    print("\n=== Google Cloud Platform Setup ===\n")
    
    # Check if gcloud is installed
    try:
        subprocess.run(["gcloud", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("gcloud CLI is not installed.")
        print("\nTo install gcloud CLI:")
        print("1. Visit: https://cloud.google.com/sdk/docs/install")
        print("2. Follow the installation instructions for your OS")
        print("3. Run this script again after installation")
        sys.exit(1)
    
    print("This will guide you through setting up Google Cloud Secret Manager.\n")
    
    # Initialize gcloud
    print("Step 1: Authenticate with Google Cloud")
    input("Press Enter to run 'gcloud auth login'...")
    subprocess.run(["gcloud", "auth", "login"])
    
    # Set up application default credentials
    print("\nStep 2: Set up Application Default Credentials")
    input("Press Enter to run 'gcloud auth application-default login'...")
    subprocess.run(["gcloud", "auth", "application-default", "login"])
    
    # Select or create project
    print("\nStep 3: Select or create a GCP project")
    create_new = input("Create a new project? (y/N): ").lower().strip() == "y"
    
    if create_new:
        project_id = input("Enter new project ID (e.g., wheel-trading-bot): ").strip()
        project_name = input("Enter project name (e.g., Wheel Trading Bot): ").strip()
        
        print(f"\nCreating project '{project_id}'...")
        result = subprocess.run([
            "gcloud", "projects", "create", project_id,
            f"--name={project_name}"
        ])
        if result.returncode != 0:
            print("Failed to create project. It may already exist.")
    else:
        # List existing projects
        print("\nExisting projects:")
        subprocess.run(["gcloud", "projects", "list"])
        project_id = input("\nEnter project ID to use: ").strip()
    
    # Set the project
    print(f"\nSetting active project to '{project_id}'...")
    subprocess.run(["gcloud", "config", "set", "project", project_id])
    
    # Enable Secret Manager API
    print("\nStep 4: Enable Secret Manager API")
    print("Enabling Secret Manager API...")
    result = subprocess.run([
        "gcloud", "services", "enable", "secretmanager.googleapis.com",
        "--project", project_id
    ])
    
    if result.returncode == 0:
        print("✓ Secret Manager API enabled successfully")
    else:
        print("Failed to enable Secret Manager API. You may need to enable billing.")
        print("Visit: https://console.cloud.google.com/billing")
    
    # Set environment variable
    print(f"\nStep 5: Set environment variable")
    print(f"Add this to your shell profile (.bashrc, .zshrc, etc.):")
    print(f"export GCP_PROJECT_ID=\"{project_id}\"")
    
    # Also set it for current session
    os.environ["GCP_PROJECT_ID"] = project_id
    
    print("\n✓ GCP setup complete!")
    print("\nNote: To use Secret Manager, you may need to:")
    print("1. Enable billing for your project")
    print("2. Install the Python client: pip install google-cloud-secret-manager")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup secrets for Unity Wheel Trading Bot"
    )
    parser.add_argument(
        "--provider",
        choices=["local", "gcp", "auto"],
        default="auto",
        help="Secret storage provider (default: auto-detect)"
    )
    parser.add_argument(
        "--setup-gcp",
        action="store_true",
        help="Run GCP setup wizard"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check current configuration"
    )
    
    args = parser.parse_args()
    
    # Handle GCP setup
    if args.setup_gcp:
        setup_gcp()
        return
    
    # Determine provider
    if args.provider == "auto":
        if check_gcp_setup():
            provider = SecretProvider.GCP
            print("\n✓ Using Google Cloud Secret Manager")
        else:
            provider = SecretProvider.LOCAL
            print("\n✓ Using local encrypted storage")
            print("\nTo use Google Cloud Secret Manager, run:")
            print("  python scripts/setup-secrets.py --setup-gcp")
    else:
        provider = SecretProvider(args.provider)
    
    # Initialize secret manager
    try:
        manager = SecretManager(provider=provider)
    except Exception as e:
        print(f"\n✗ Failed to initialize secret manager: {e}")
        if provider == SecretProvider.GCP:
            print("\nTry running: python scripts/setup-secrets.py --setup-gcp")
        sys.exit(1)
    
    # Check configuration
    if args.check_only:
        print("\n=== Current Configuration ===")
        configured = manager.list_configured_services()
        for service, is_configured in configured.items():
            status = "✓" if is_configured else "✗"
            print(f"{status} {service}")
        return
    
    # Run interactive setup
    manager.setup_all_credentials()
    
    # Show how to use in code
    print("\n=== Usage Examples ===\n")
    print("# In your Python code:")
    print("from src.unity_wheel.secrets import SecretManager")
    print()
    print("# Initialize (auto-detects provider)")
    print("secrets = SecretManager()")
    print()
    print("# Get Schwab credentials")
    print("schwab_creds = secrets.get_credentials('schwab')")
    print("client_id = schwab_creds['client_id']")
    print("client_secret = schwab_creds['client_secret']")
    print()
    print("# Get individual secret")
    print("databento_key = secrets.get_secret('databento_api_key')")
    
    # Environment-specific instructions
    if provider == SecretProvider.LOCAL:
        print(f"\nSecrets stored locally at: ~/.wheel_trading/secrets/")
        print("These are encrypted using machine-specific keys.")
    else:
        project_id = os.environ.get("GCP_PROJECT_ID", "<your-project-id>")
        print(f"\nSecrets stored in GCP project: {project_id}")
        print("View in console: https://console.cloud.google.com/security/secret-manager")
    
    # Ask if user wants to test credentials
    print("\n" + "=" * 60)
    test_now = input("\nWould you like to test the credentials now? (y/N): ").lower().strip() == "y"
    if test_now:
        print("\nStarting credential tests...")
        result = subprocess.run([
            sys.executable, 
            str(Path(__file__).parent / "test-secrets.py")
        ])
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()