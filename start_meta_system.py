#!/usr/bin/env python3
"""
Meta System Launcher - Actually starts and runs the complete meta system
"""

import asyncio
import signal
import sys

from meta_coordinator import start_meta_coordination
from meta_daemon import MetaDaemon
from meta_prime import MetaPrime
from meta_reality_bridge import MetaRealityBridge


class MetaSystemRunner:
    """Runs the complete integrated meta system"""

    def __init__(self):
        self.meta_prime = None
        self.reality_bridge = None
        self.daemon = None
        self.running = False

    async def start_complete_system(self):
        """Start all components and run integrated system"""

        print("🚀 Starting Complete Meta System...")

        # 1. Initialize MetaPrime
        print("  🌱 Initializing MetaPrime...")
        self.meta_prime = MetaPrime()

        # 2. Start Reality Bridge (connects to real development)
        print("  🌉 Starting Reality Bridge...")
        self.reality_bridge = MetaRealityBridge()
        self.reality_bridge.start_reality_observation()

        # 3. Start Meta Daemon (quality enforcement)
        print("  🤖 Starting Meta Daemon...")
        self.daemon = MetaDaemon()
        daemon_task = asyncio.create_task(self.daemon.start())

        # 4. Start Meta Coordination (evolution system)
        print("  🧠 Starting Meta Coordination...")
        coord_task = asyncio.create_task(start_meta_coordination())

        print("✅ All Meta System Components Started!")
        print("\n📊 System Status:")
        print(
            f"  • MetaPrime: Active ({self.meta_prime.get_observation_count()} observations)"
        )
        print("  • Reality Bridge: Monitoring files")
        print("  • Meta Daemon: Quality enforcement active")
        print("  • Meta Coordinator: Evolution monitoring active")

        self.running = True

        # Run until interrupted
        try:
            await asyncio.gather(daemon_task, coord_task)
        except asyncio.CancelledError:
            print("\n🛑 Shutting down Meta System...")
            await self.shutdown()

    async def shutdown(self):
        """Gracefully shutdown all components"""

        if self.reality_bridge:
            self.reality_bridge.shutdown()

        if self.daemon:
            await self.daemon._shutdown()

        print("✅ Meta System shutdown complete")
        self.running = False


async def main():
    """Main entry point"""

    runner = MetaSystemRunner()

    # Handle shutdown signals
    def signal_handler():
        print("\n📡 Received shutdown signal...")
        if runner.running:
            asyncio.create_task(runner.shutdown())

    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, lambda s, f: signal_handler())

    try:
        await runner.start_complete_system()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        await runner.shutdown()


if __name__ == "__main__":
    print("🔥 Meta System - Complete Integration")
    print("=" * 50)

    if len(sys.argv) > 1:
        if sys.argv[1] == "--status":
            # Quick status check
            try:
                meta = MetaPrime()
                print(f"📊 {meta.status_report()}")
                print(f"🔄 Evolution ready: {meta.should_evolve()}")
            except Exception as e:
                print(f"❌ System not running: {e}")
        elif sys.argv[1] == "--evolve":
            # Trigger evolution
            meta = MetaPrime()
            if meta.should_evolve():
                success = meta.evolve()
                print(f"🧬 Evolution {'successful' if success else 'failed'}")
            else:
                print("⏳ System not ready to evolve yet")
        else:
            print("Usage: python start_meta_system.py [--status|--evolve]")
    else:
        # Start full system
        asyncio.run(main())
