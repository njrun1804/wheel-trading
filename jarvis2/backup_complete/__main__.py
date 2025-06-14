"""Main entry point for Jarvis2 when run as a module."""
from jarvis2.cli import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())