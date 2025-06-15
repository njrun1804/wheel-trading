#!/usr/bin/env python3
"""
Launch Claude-Meta Integration System
Complete production system for real-time Claude thought monitoring
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any
import signal

from meta_claude_integration_hooks import MetaClaudeIntegrationManager
from claude_stream_integration import ClaudeThoughtStreamIntegration
from meta_prime import MetaPrime


class ClaudeMetaIntegrationSystem:
    """Complete Claude-Meta integration system"""
    
    def __init__(self):
        self.integration_manager = None
        self.running = False
        self.start_time = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.running = False
        if self.integration_manager:
            self.integration_manager.stop_monitoring()
        sys.exit(0)
    
    async def start_system(self, api_key: str = None):
        """Start the complete Claude-Meta integration system"""
        
        print("🚀 CLAUDE-META INTEGRATION SYSTEM")
        print("=" * 50)
        print("🧠 Real-time Claude thought monitoring")
        print("🔗 Meta system evolutionary learning")
        print("🎯 Complete autonomous development")
        print()
        
        # Validate API key
        if not api_key:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            
        if not api_key:
            print("❌ ANTHROPIC_API_KEY required")
            print("   Set environment variable or pass as argument")
            return False
        
        try:
            # Initialize integration manager
            self.integration_manager = MetaClaudeIntegrationManager()
            self.start_time = asyncio.get_event_loop().time()
            self.running = True
            
            # Start integrated monitoring
            await self.integration_manager.start_integrated_monitoring(api_key)
            
            print("✅ Claude-Meta integration system started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start integration system: {e}")
            return False
    
    async def run_interactive_session(self, api_key: str = None):
        """Run interactive session for testing and demonstration"""
        
        success = await self.start_system(api_key)
        if not success:
            return
        
        print("\n🎮 INTERACTIVE MODE")
        print("Commands:")
        print("  'test' - Run with sample trading requests")
        print("  'message <text>' - Process single message")
        print("  'status' - Show system status")
        print("  'analytics' - Show detailed analytics")
        print("  'quit' - Exit system")
        print()
        
        while self.running:
            try:
                user_input = input("claude-meta> ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'test':
                    await self._run_test_requests()
                elif user_input.lower() == 'status':
                    self._show_status()
                elif user_input.lower() == 'analytics':
                    self._show_analytics()
                elif user_input.startswith('message '):
                    message = user_input[8:]  # Remove 'message ' prefix
                    await self._process_single_message(message)
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break
        
        self._signal_handler(signal.SIGINT, None)
    
    async def _run_test_requests(self):
        """Run test requests through the system"""
        
        print("🧪 Running test requests...")
        
        test_requests = [
            "Analyze my wheel trading strategy and suggest optimizations for current market volatility",
            "Help me implement dynamic position sizing based on VIX levels and portfolio heat",
            "Create a systematic approach to rolling puts in my wheel strategy when assignments occur",
            "Optimize my options portfolio for maximum theta capture while controlling gamma risk"
        ]
        
        if not self.integration_manager.claude_integration:
            print("❌ Claude integration not started")
            return
        
        monitor = self.integration_manager.claude_monitor
        
        for i, request in enumerate(test_requests, 1):
            print(f"\n📝 Test {i}/4: {request[:60]}...")
            
            try:
                claude_request = await monitor.stream_with_thinking(request, thinking_budget=8000)
                
                print(f"✅ Completed:")
                print(f"   Thinking tokens: {claude_request.total_thinking_tokens}")
                print(f"   Patterns detected: {claude_request.patterns_detected}")
                print(f"   Duration: {claude_request.completion_time - claude_request.start_time:.2f}s")
                
            except Exception as e:
                print(f"❌ Error processing request: {e}")
        
        print(f"\n🎯 Test completed - check analytics for insights generated")
    
    async def _process_single_message(self, message: str):
        """Process a single message through the system"""
        
        if not self.integration_manager.claude_integration:
            print("❌ Claude integration not started")
            return
        
        print(f"📝 Processing: {message[:80]}...")
        
        try:
            monitor = self.integration_manager.claude_monitor
            claude_request = await monitor.stream_with_thinking(message)
            
            print(f"✅ Processed successfully:")
            print(f"   Thinking tokens: {claude_request.total_thinking_tokens}")
            print(f"   Thinking deltas: {len(claude_request.thinking_deltas)}")
            print(f"   Patterns: {claude_request.patterns_detected}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def _show_status(self):
        """Show current system status"""
        
        if not self.integration_manager:
            print("❌ Integration manager not initialized")
            return
        
        status = self.integration_manager.get_integration_status()
        
        print("\n📊 SYSTEM STATUS")
        print("-" * 30)
        print(f"🔄 Active monitoring: {status['active_monitoring']}")
        print(f"🧠 Claude integration: {status['claude_integration_active']}")
        print(f"💡 Insights generated: {status['insights_generated']}")
        print(f"🧬 Meta evolutions: {status['meta_evolutions_triggered']}")
        print(f"🧵 Pattern cache: {status['pattern_cache_size']} patterns")
        
        if status['recent_insights']:
            print(f"\n🔍 Recent insights:")
            for insight in status['recent_insights']:
                print(f"   • {insight['type']} (confidence: {insight['confidence']:.2f})")
    
    def _show_analytics(self):
        """Show detailed system analytics"""
        
        if not self.integration_manager or not self.integration_manager.claude_integration:
            print("❌ System not running")
            return
        
        claude_analytics = self.integration_manager.claude_monitor.get_monitoring_analytics()
        
        print("\n📈 DETAILED ANALYTICS")
        print("=" * 40)
        
        print(f"🧠 Claude Analytics:")
        for key, value in claude_analytics.items():
            print(f"   {key}: {value}")
        
        print(f"\n🔗 Integration Analytics:")
        status = self.integration_manager.get_integration_status()
        print(f"   Insights generated: {status['insights_generated']}")
        print(f"   Meta evolutions triggered: {status['meta_evolutions_triggered']}")
        print(f"   Pattern types detected: {claude_analytics.get('recent_pattern_types', [])}")
        
        if self.start_time:
            uptime = asyncio.get_event_loop().time() - self.start_time
            print(f"   System uptime: {uptime:.1f} seconds")


async def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Claude-Meta Integration System")
    parser.add_argument("--api-key", type=str, help="Anthropic API key")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--test", action="store_true", help="Run test requests and exit")
    parser.add_argument("--message", type=str, help="Process single message and exit")
    
    args = parser.parse_args()
    
    # Check for API key
    api_key = args.api_key or os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ANTHROPIC_API_KEY required")
        print("   Set environment variable: export ANTHROPIC_API_KEY='your-key'")
        print("   Or use --api-key argument")
        return
    
    system = ClaudeMetaIntegrationSystem()
    
    try:
        if args.interactive:
            await system.run_interactive_session(api_key)
        elif args.test:
            success = await system.start_system(api_key)
            if success:
                await system._run_test_requests()
        elif args.message:
            success = await system.start_system(api_key)
            if success:
                await system._process_single_message(args.message)
        else:
            print("Use --interactive, --test, or --message options")
            print("Run with --help for more information")
            
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except Exception as e:
        print(f"❌ System error: {e}")


if __name__ == "__main__":
    asyncio.run(main())