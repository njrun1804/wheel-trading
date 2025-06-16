#!/usr/bin/env python3
"""
Error Handling Integration Example

Demonstrates how to integrate the comprehensive error handling system
into Einstein and Bolt applications with user-friendly error reporting.
"""

import asyncio
import logging
from pathlib import Path

# Configure logging to see error handling in action
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def demonstrate_basic_error_handling():
    """Demonstrate basic error handling with user-friendly messages."""
    print("\n🔍 Demonstrating Basic Error Handling")
    print("-" * 50)
    
    try:
        from error_handling_interface import handle_user_error
        
        # Simulate various types of errors that might occur
        test_errors = [
            (ValueError("Invalid search query format"), "performing search"),
            (MemoryError("Insufficient memory for large index"), "building search index"),
            (FileNotFoundError("Configuration file missing"), "loading configuration"),
            (ConnectionError("Unable to connect to external service"), "fetching data"),
            (ImportError("Required module not found"), "initializing component"),
        ]
        
        for error, user_action in test_errors:
            print(f"\n📋 Handling: {type(error).__name__}")
            
            # Handle the error with user-friendly feedback
            user_error = await handle_user_error(
                error,
                user_action=user_action,
                context={'demo_mode': True}
            )
            
            # Display user-friendly information
            print(f"🏷️  Title: {user_error.title}")
            print(f"💬 Message: {user_error.message}")
            print(f"📊 Level: {user_error.level.value}")
            print(f"🔧 Component: {user_error.component}")
            print(f"🆔 Error Code: {user_error.error_code}")
            print(f"🔄 Recovery Status: {user_error.recovery_status}")
            print(f"⚡ System Impact: {user_error.system_impact}")
            
            if user_error.suggestions:
                print("💡 Suggestions:")
                for i, suggestion in enumerate(user_error.suggestions[:3], 1):
                    print(f"   {i}. {suggestion}")
            
            print("✅ Error handled successfully")
    
    except Exception as e:
        print(f"❌ Failed to demonstrate error handling: {e}")


async def demonstrate_system_status():
    """Demonstrate system status monitoring."""
    print("\n📊 Demonstrating System Status Monitoring")
    print("-" * 50)
    
    try:
        from error_handling_interface import get_user_system_status
        
        # Get current system status
        status = await get_user_system_status()
        
        print(f"🏥 Overall Health: {status.overall_health}")
        print(f"📉 Degradation Level: {status.degradation_level}")
        print(f"⏱️  Uptime: {status.uptime_hours:.2f} hours")
        print(f"🎯 Recovery Rate: {status.recovery_rate:.1f}%")
        print(f"📈 Performance: {status.performance_status}")
        print(f"🚨 Recent Errors: {status.recent_errors}")
        
        if status.component_status:
            print("\n🔧 Component Status:")
            for component, comp_status in status.component_status.items():
                print(f"   • {component}: {comp_status}")
        
        if status.recommendations:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(status.recommendations[:5], 1):
                print(f"   {i}. {rec}")
        
        print("✅ System status retrieved successfully")
    
    except Exception as e:
        print(f"❌ Failed to get system status: {e}")


async def demonstrate_error_scenarios():
    """Demonstrate handling of specific error scenarios."""
    print("\n🎭 Demonstrating Specific Error Scenarios")
    print("-" * 50)
    
    scenarios = [
        {
            'name': 'Search Index Corruption',
            'error': RuntimeError("FAISS index file corrupted"),
            'action': 'searching for code patterns',
            'context': {'index_size': 50000, 'corruption_detected': True}
        },
        {
            'name': 'GPU Memory Exhaustion',
            'error': RuntimeError("CUDA out of memory"),
            'action': 'processing embeddings',
            'context': {'gpu_memory_used': '95%', 'batch_size': 1000}
        },
        {
            'name': 'File Watcher Failure',
            'error': OSError("Too many open files"),
            'action': 'monitoring file changes',
            'context': {'open_files': 2048, 'watch_paths': 50}
        },
        {
            'name': 'Database Connection Lost',
            'error': ConnectionError("Database connection timeout"),
            'action': 'storing search results',
            'context': {'retry_count': 3, 'timeout_seconds': 30}
        }
    ]
    
    try:
        from error_handling_interface import handle_user_error
        
        for scenario in scenarios:
            print(f"\n🎬 Scenario: {scenario['name']}")
            
            user_error = await handle_user_error(
                scenario['error'],
                user_action=scenario['action'],
                context=scenario['context']
            )
            
            print(f"   📋 User sees: {user_error.title}")
            print(f"   💬 Message: {user_error.message}")
            print(f"   🔄 Recovery: {user_error.recovery_status}")
            print(f"   💡 Key suggestion: {user_error.suggestions[0] if user_error.suggestions else 'None'}")
            
        print("\n✅ All scenarios handled successfully")
    
    except Exception as e:
        print(f"❌ Failed to demonstrate scenarios: {e}")


def demonstrate_help_system():
    """Demonstrate the help system."""
    print("\n❓ Demonstrating Help System")
    print("-" * 50)
    
    try:
        from error_handling_interface import get_help_for_error
        
        # Get general help
        print("📖 General Help (first 300 chars):")
        general_help = get_help_for_error()
        print(general_help[:300] + "...")
        
        # Get specific error help
        print("\n🔍 Help for Memory Errors:")
        memory_help = get_help_for_error("MEMORY_ERROR")
        lines = memory_help.split('\n')
        for line in lines[:10]:  # Show first 10 lines
            if line.strip():
                print(f"   {line}")
        
        print("\n✅ Help system working correctly")
    
    except Exception as e:
        print(f"❌ Failed to demonstrate help system: {e}")


def demonstrate_error_report():
    """Demonstrate error report generation."""
    print("\n📄 Demonstrating Error Report Generation")
    print("-" * 50)
    
    try:
        from error_handling_interface import export_user_error_report
        
        # Generate error report
        report_path = export_user_error_report()
        
        print(f"📊 Error report generated: {report_path}")
        print(f"📁 File size: {report_path.stat().st_size} bytes")
        
        # Show a snippet of the report
        import json
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        print("\n📋 Report contents (sample):")
        print(f"   • Timestamp: {report_data.get('timestamp_str', 'Unknown')}")
        print(f"   • Project root: {report_data.get('project_root', 'Unknown')}")
        print(f"   • Uptime: {report_data.get('uptime_hours', 0):.2f} hours")
        
        if 'components_available' in report_data:
            print("   • Available components:")
            for comp, available in report_data['components_available'].items():
                status = "✅" if available else "❌"
                print(f"     {status} {comp}")
        
        # Cleanup (optional - remove in production)
        report_path.unlink()
        print("\n✅ Error report generated successfully")
    
    except Exception as e:
        print(f"❌ Failed to generate error report: {e}")


async def demonstrate_integration_points():
    """Demonstrate integration with Einstein and Bolt systems."""
    print("\n🔗 Demonstrating System Integration")
    print("-" * 50)
    
    # Test Bolt integration
    try:
        from bolt.error_handling.integration import get_integrated_error_manager
        
        manager = get_integrated_error_manager()
        health_summary = manager.get_system_health_summary()
        
        print("🔧 Bolt Integration:")
        print(f"   • Status: {health_summary.get('system_status', 'unknown')}")
        print(f"   • Recent errors: {health_summary.get('total_errors_last_hour', 0)}")
        print(f"   • Recovery rate: {health_summary.get('recovery_success_rate', 0):.1f}%")
        print("   ✅ Bolt integration working")
        
    except ImportError:
        print("🔧 Bolt Integration: ⚠️  Components not available")
    except Exception as e:
        print(f"🔧 Bolt Integration: ❌ Error - {e}")
    
    # Test Einstein integration
    try:
        from einstein.error_handling.diagnostics import get_einstein_diagnostics
        
        diagnostics = get_einstein_diagnostics()
        quick_status = await diagnostics.health_checker.quick_health_check()
        
        print(f"🧠 Einstein Integration:")
        print(f"   • Health status: {quick_status.value}")
        print("   ✅ Einstein integration working")
        
    except ImportError:
        print("🧠 Einstein Integration: ⚠️  Components not available")
    except Exception as e:
        print(f"🧠 Einstein Integration: ❌ Error - {e}")


async def main():
    """Run all demonstrations."""
    print("🚀 Error Handling System Demonstration")
    print("=" * 80)
    print("This example shows how to integrate comprehensive error handling")
    print("with user-friendly messages, system monitoring, and recovery.")
    print("=" * 80)
    
    # Run all demonstrations
    demonstrations = [
        demonstrate_basic_error_handling,
        demonstrate_system_status,
        demonstrate_error_scenarios,
        demonstrate_help_system,
        demonstrate_error_report,
        demonstrate_integration_points,
    ]
    
    for demo in demonstrations:
        try:
            if asyncio.iscoroutinefunction(demo):
                await demo()
            else:
                demo()
        except Exception as e:
            print(f"❌ Demonstration {demo.__name__} failed: {e}")
        
        print()  # Add spacing between demonstrations
    
    print("🎉 Demonstration Complete!")
    print("=" * 80)
    print("Key Benefits of This Error Handling System:")
    print("• 🎯 User-friendly error messages instead of technical stack traces")
    print("• 🔄 Automatic error recovery and fallback mechanisms")
    print("• 📊 System health monitoring and degradation management")
    print("• 🛡️ Graceful degradation to keep system functional")
    print("• 📄 Comprehensive error reporting for debugging")
    print("• 🤝 Unified handling across Einstein and Bolt systems")
    print("• 💡 Helpful suggestions and troubleshooting guidance")
    print()
    print("Integration Tips:")
    print("• Use handle_user_error() for any exception that users might see")
    print("• Check get_user_system_status() for system health monitoring")
    print("• Use export_user_error_report() for debugging and support")
    print("• Call get_help_for_error() to provide contextual help")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())