#!/usr/bin/env python3
"""
Unified Assessment Engine Demo

Demonstrates the unified assessment engine with various natural language commands.
Shows the complete pipeline from context gathering through execution.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from unity_wheel.unified_assessment import UnifiedAssessmentEngine
from unity_wheel.unified_assessment.schemas.command import CommandStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_command(engine: UnifiedAssessmentEngine, command: str, description: str):
    """Demonstrate processing a single command."""
    
    print(f"\n{'='*80}")
    print(f"🎯 DEMO: {description}")
    print(f"📝 Command: '{command}'")
    print(f"{'='*80}")
    
    start_time = time.perf_counter()
    
    try:
        # Process the command
        result = await engine.process_command(command)
        
        duration = time.perf_counter() - start_time
        
        # Display results
        print(f"\n✅ Status: {result.status.value}")
        print(f"🎯 Success: {result.success}")
        print(f"⏱️  Duration: {duration:.2f}s")
        print(f"📊 Intent Confidence: {result.metrics.intent_confidence:.2f}")
        print(f"🔍 Context Confidence: {result.metrics.context_confidence:.2f}")
        
        if result.summary:
            print(f"\n📋 Summary: {result.summary}")
        
        if result.findings:
            print(f"\n🔍 Findings:")
            for i, finding in enumerate(result.findings[:5], 1):
                print(f"  {i}. {finding}")
        
        if result.recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(result.recommendations[:5], 1):
                print(f"  {i}. {rec}")
        
        if result.actions_taken:
            print(f"\n🚀 Actions Taken:")
            for i, action in enumerate(result.actions_taken[:3], 1):
                print(f"  {i}. {action}")
        
        if result.files_affected:
            print(f"\n📁 Files Affected: {len(result.files_affected)}")
            for file_path in result.files_affected[:3]:
                print(f"  • {file_path}")
        
        if result.errors:
            print(f"\n❌ Errors ({len(result.errors)}):")
            for error in result.errors[:3]:
                print(f"  • {error.error_type}: {error.error_message}")
        
        if result.warnings:
            print(f"\n⚠️  Warnings ({len(result.warnings)}):")
            for warning in result.warnings[:3]:
                print(f"  • {warning}")
        
        # Performance metrics
        print(f"\n📈 Performance Metrics:")
        print(f"  • Context Gathering: {result.metrics.context_gathering_ms:.1f}ms")
        print(f"  • Intent Analysis: {result.metrics.intent_analysis_ms:.1f}ms")
        print(f"  • Action Planning: {result.metrics.action_planning_ms:.1f}ms")
        print(f"  • Execution: {result.metrics.execution_ms:.1f}ms")
        print(f"  • Tasks Completed: {result.metrics.tasks_completed}")
        print(f"  • Tasks Failed: {result.metrics.tasks_failed}")
        
        return result
        
    except Exception as e:
        duration = time.perf_counter() - start_time
        print(f"\n❌ Error processing command: {e}")
        print(f"⏱️  Duration: {duration:.2f}s")
        return None


async def run_demo_suite():
    """Run complete demo suite with various command types."""
    
    print("🚀 Unified Assessment Engine Demo")
    print("=" * 80)
    
    # Initialize the engine
    print("\n🔧 Initializing Unified Assessment Engine...")
    engine = UnifiedAssessmentEngine({
        "context": {
            "max_files": 20,
            "search_depth": 2
        },
        "intent": {
            "confidence_threshold": 0.6
        },
        "planning": {
            "optimization_target": "balanced"
        },
        "routing": {
            "prefer_parallel": True
        }
    })
    
    try:
        await engine.initialize()
        print("✅ Engine initialized successfully")
        
        # Demo commands covering different intent categories
        demo_commands = [
            (
                "fix authentication issue",
                "FIX: Debugging and fixing an authentication problem"
            ),
            (
                "create new trading strategy for Unity stock",
                "CREATE: Building a new trading strategy component"
            ),
            (
                "optimize performance of the wheel strategy",
                "OPTIMIZE: Performance optimization task"
            ),
            (
                "analyze risk management components",
                "ANALYZE: Code analysis and review"
            ),
            (
                "refactor the position sizing logic",
                "REFACTOR: Code restructuring task"
            ),
            (
                "show me all authentication related files",
                "QUERY: Information retrieval task"
            ),
            (
                "test the backtesting system",
                "TEST: Testing and validation task"
            ),
            (
                "find performance bottlenecks in the trading system",
                "COMPLEX: Multi-step analysis and optimization"
            ),
        ]
        
        results = []
        
        # Process each demo command
        for command, description in demo_commands:
            result = await demo_command(engine, command, description)
            if result:
                results.append(result)
            
            # Short pause between demos
            await asyncio.sleep(0.5)
        
        # Summary statistics
        print(f"\n{'='*80}")
        print("📊 DEMO SUMMARY")
        print(f"{'='*80}")
        
        successful_commands = sum(1 for r in results if r.success)
        total_commands = len(results)
        
        print(f"📝 Total Commands: {total_commands}")
        print(f"✅ Successful: {successful_commands}")
        print(f"❌ Failed: {total_commands - successful_commands}")
        print(f"📈 Success Rate: {successful_commands/total_commands:.1%}")
        
        if results:
            avg_duration = sum(r.get_duration_seconds() for r in results) / len(results)
            avg_context_time = sum(r.metrics.context_gathering_ms for r in results) / len(results)
            avg_intent_time = sum(r.metrics.intent_analysis_ms for r in results) / len(results)
            avg_planning_time = sum(r.metrics.action_planning_ms for r in results) / len(results)
            avg_execution_time = sum(r.metrics.execution_ms for r in results) / len(results)
            
            print(f"\n⏱️  Average Timings:")
            print(f"  • Total Duration: {avg_duration:.2f}s")
            print(f"  • Context Gathering: {avg_context_time:.1f}ms")
            print(f"  • Intent Analysis: {avg_intent_time:.1f}ms")
            print(f"  • Action Planning: {avg_planning_time:.1f}ms")
            print(f"  • Execution: {avg_execution_time:.1f}ms")
            
            # Confidence scores
            avg_intent_confidence = sum(r.metrics.intent_confidence for r in results) / len(results)
            avg_context_confidence = sum(r.metrics.context_confidence for r in results) / len(results)
            
            print(f"\n🎯 Average Confidence Scores:")
            print(f"  • Intent Confidence: {avg_intent_confidence:.2f}")
            print(f"  • Context Confidence: {avg_context_confidence:.2f}")
        
        # Engine statistics
        engine_stats = await engine.get_engine_stats()
        print(f"\n🔧 Engine Statistics:")
        print(f"  • Engine ID: {engine_stats['engine_id']}")
        print(f"  • Commands Processed: {engine_stats['total_commands_processed']}")
        print(f"  • Success Rate: {engine_stats['success_rate']:.1%}")
        print(f"  • Avg Processing Time: {engine_stats['average_processing_time_ms']:.1f}ms")
        
        # Export detailed results for analysis
        export_results(results, engine_stats)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
    
    finally:
        # Cleanup
        print(f"\n🔄 Shutting down engine...")
        await engine.shutdown()
        print("✅ Demo complete")


def export_results(results, engine_stats):
    """Export demo results to JSON for analysis."""
    
    try:
        export_data = {
            "demo_timestamp": time.time(),
            "engine_stats": engine_stats,
            "results": [result.to_dict() for result in results],
            "summary": {
                "total_commands": len(results),
                "successful_commands": sum(1 for r in results if r.success),
                "average_duration_seconds": sum(r.get_duration_seconds() for r in results) / len(results) if results else 0,
                "average_confidence": {
                    "intent": sum(r.metrics.intent_confidence for r in results) / len(results) if results else 0,
                    "context": sum(r.metrics.context_confidence for r in results) / len(results) if results else 0
                }
            }
        }
        
        export_file = Path(__file__).parent / "unified_assessment_demo_results.json"
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"\n📊 Results exported to: {export_file}")
        
    except Exception as e:
        logger.warning(f"Failed to export results: {e}")


async def interactive_demo():
    """Run interactive demo where user can enter commands."""
    
    print("\n🎮 INTERACTIVE MODE")
    print("Enter natural language commands to see the unified assessment engine in action.")
    print("Type 'quit' to exit.\n")
    
    engine = UnifiedAssessmentEngine()
    await engine.initialize()
    
    try:
        while True:
            try:
                command = input("🎯 Enter command: ").strip()
                
                if command.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not command:
                    continue
                
                result = await demo_command(engine, command, "Interactive Command")
                
            except KeyboardInterrupt:
                print("\n👋 Exiting interactive mode...")
                break
            except EOFError:
                break
    
    finally:
        await engine.shutdown()


async def main():
    """Main demo entry point."""
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        await interactive_demo()
    else:
        await run_demo_suite()


if __name__ == "__main__":
    asyncio.run(main())