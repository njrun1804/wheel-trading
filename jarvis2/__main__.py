"""Main entry point for Jarvis2 when run as a module."""
import asyncio

from jarvis2.cli import main

if __name__ == "__main__":
    asyncio.run(main())