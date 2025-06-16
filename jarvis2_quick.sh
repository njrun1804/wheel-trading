#!/bin/bash
# Quick launcher for Jarvis2 with a query
python3 -c "
import asyncio
from jarvis2.core.orchestrator import Jarvis2Orchestrator, CodeRequest

async def main():
    jarvis = Jarvis2Orchestrator()
    await jarvis.initialize()
    
    query = '$1'
    if not query:
        query = 'Create a function to calculate fibonacci numbers'
    
    print(f'\\n🤖 Jarvis2: Generating code for: {query}\\n')
    
    request = CodeRequest(query)
    solution = await jarvis.generate_code(request)
    
    print('Generated code:')
    print('='*60)
    print(solution.code)
    print('='*60)
    print(f'\\nConfidence: {solution.confidence:.0%}')
    print(f'Time: {solution.metrics[\"generation_time_ms\"]:.0f}ms')
    
    await jarvis.shutdown()

asyncio.run(main())
"
