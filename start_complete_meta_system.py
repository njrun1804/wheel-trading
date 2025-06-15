#!/usr/bin/env python3
"""
Complete Meta System Startup - Actually start everything integrated
This is what you run to get the ACTUAL working meta-system as described
"""

import asyncio
import sys
from pathlib import Path

# Import the integrated components
from meta_loop_integrated import start_integrated_meta_loop
from claude_code_integration import start_claude_code_monitoring
from mistake_detection import run_mistake_detection_on_project


def show_startup_menu():
    """Show what the meta-system can actually do"""
    
    print("🔥 Complete Meta-System - Choose Your Adventure")
    print("=" * 60)
    print()
    print("🎯 ACTUALLY WORKING OPTIONS:")
    print()
    print("1. 🔄 Start Integrated Meta-Loop")
    print("   → Monitors Claude Code, detects mistakes, triggers evolution")
    print("   → This is the ACTUAL meta-system as described")
    print()
    print("2. 👁️  Start Claude Code Monitoring Only") 
    print("   → Just monitor what Claude does and collect feedback")
    print("   → Good for testing without full meta-loop")
    print()
    print("3. 🕵️  Run Mistake Detection on Project")
    print("   → Check all Python files for potential issues")
    print("   → One-time scan, no continuous monitoring")
    print()
    print("4. 📊 Quick System Status")
    print("   → Show current meta-system status")
    print("   → See observations, evolution readiness, etc.")
    print()
    print("5. 🧬 Trigger Manual Evolution")
    print("   → Force the meta-system to evolve now")
    print("   → See self-modification in action")
    print()
    print("6. ❌ Exit")
    print()


async def run_integrated_meta_loop():
    """Start the complete integrated meta-loop"""
    print("🚀 Starting Complete Integrated Meta-Loop...")
    print("This monitors Claude Code, detects mistakes, and triggers evolution automatically.")
    print("Press Ctrl+C to stop.")
    print()
    
    await start_integrated_meta_loop()


def run_claude_monitoring_only():
    """Start just Claude Code monitoring"""
    print("👁️  Starting Claude Code Monitoring...")
    print("This will watch your file edits and prompt for feedback.")
    print("Press Ctrl+C to stop.")
    print()
    
    start_claude_code_monitoring()


def run_mistake_detection():
    """Run mistake detection on the project"""
    print("🕵️  Running Project-Wide Mistake Detection...")
    print()
    
    results = run_mistake_detection_on_project()
    
    print(f"\n✅ Mistake detection complete!")
    return results


def show_system_status():
    """Show current meta-system status"""
    print("📊 Meta-System Status Check...")
    print()
    
    try:
        from meta_prime import MetaPrime
        meta = MetaPrime()
        
        print("🔬 MetaPrime Status:")
        print(f"   Observations: {meta.get_observation_count():,}")
        print(f"   Evolution ready: {'✅ Yes' if meta.should_evolve() else '❌ No'}")
        
        # Get recent activity
        recent = meta.get_recent_observations(5)
        print(f"   Recent activity:")
        for timestamp, event_type, details, context in recent:
            age = time.time() - timestamp
            print(f"     • {event_type} ({age:.0f}s ago)")
            
        # Check if system has evolved
        evolution_events = [obs for obs in recent if 'evolution' in obs[1]]
        if evolution_events:
            print(f"   🧬 Recent evolution activity: {len(evolution_events)} events")
        else:
            print(f"   ⏳ No recent evolution activity")
            
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        
    print()


def trigger_manual_evolution():
    """Trigger evolution manually"""
    print("🧬 Triggering Manual Evolution...")
    print()
    
    try:
        import time
        from meta_prime import MetaPrime
        meta = MetaPrime()
        
        print(f"Current observations: {meta.get_observation_count():,}")
        print(f"Evolution ready: {meta.should_evolve()}")
        
        if meta.should_evolve():
            print("🚀 Triggering evolution...")
            success = meta.evolve()
            
            if success:
                print("✅ Evolution successful!")
                print("🔍 Check the meta_prime.py file for changes")
                
                # Show what changed
                recent = meta.get_recent_observations(3)
                for timestamp, event_type, details, context in recent:
                    if 'evolution' in event_type:
                        print(f"   📝 {event_type}: {details}")
            else:
                print("❌ Evolution failed")
        else:
            print("⏳ System not ready to evolve yet")
            print("   Need more observations or different conditions")
            
    except Exception as e:
        print(f"❌ Error during evolution: {e}")
        
    print()


async def main():
    """Main menu loop"""
    
    while True:
        show_startup_menu()
        
        try:
            choice = input("Enter your choice (1-6): ").strip()
            print()
            
            if choice == "1":
                await run_integrated_meta_loop()
            elif choice == "2":
                run_claude_monitoring_only()
            elif choice == "3":
                run_mistake_detection()
            elif choice == "4":
                show_system_status()
            elif choice == "5":
                trigger_manual_evolution()
            elif choice == "6":
                print("👋 Goodbye!")
                break
            else:
                print("❓ Invalid choice. Please enter 1-6.")
                
            # Pause before showing menu again
            if choice in ["1", "2"]:
                break  # These are continuous, exit after
            else:
                input("\nPress Enter to continue...")
                print()
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    import time
    
    print("🚀 Complete Meta-System Startup")
    print("Finally - the ACTUAL integrated meta-system that works as described!")
    print()
    
    # Check dependencies
    required_files = [
        "meta_prime.py",
        "meta_loop_integrated.py", 
        "claude_code_integration.py",
        "mistake_detection.py"
    ]
    
    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        print(f"❌ Missing required files: {missing}")
        print("Make sure all meta-system components are in the current directory.")
        sys.exit(1)
        
    print("✅ All components found. Starting menu...")
    print()
    
    # META COMPLETELY DISABLED FOR EINSTEIN TESTING
    print("🔪 Complete meta system DISABLED for clean Einstein testing")
    exit(0)
    
    # try:
    #     asyncio.run(main())
    # except KeyboardInterrupt:
    #     print("\n👋 Goodbye!")
    # except Exception as e:
    #     print(f"❌ Startup error: {e}")
    #     sys.exit(1)