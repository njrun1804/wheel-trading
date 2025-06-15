#!/usr/bin/env python3
"""
How to use hardware-accelerated sequential thinking for real tasks.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.unity_wheel.accelerated_tools.sequential_thinking_mac_optimized import get_mac_optimized_thinking


async def use_for_coding_task():
    """Example: Use sequential thinking for complex coding decisions."""
    thinking = get_mac_optimized_thinking()
    
    steps = await thinking.think(
        goal="Refactor a legacy codebase to use async/await patterns",
        constraints=[
            "Maintain backwards compatibility",
            "Minimize breaking changes",
            "Improve performance by 50%",
            "Keep code readable"
        ],
        max_steps=20
    )
    
    print("CODING TASK SOLUTION:")
    print("=" * 50)
    for step in steps:
        print(f"\n{step.step_number}. {step.action}")
        print(f"   Why: {step.reasoning}")
        print(f"   Confidence: {step.confidence:.2%}")
    
    thinking.close()
    

async def use_for_architecture_design():
    """Example: Design system architecture."""
    thinking = get_mac_optimized_thinking()
    
    steps = await thinking.think(
        goal="Design a distributed ML training system",
        constraints=[
            "Scale to 1000 GPUs",
            "Handle node failures gracefully", 
            "Minimize communication overhead",
            "Support multiple frameworks (PyTorch, JAX)",
            "Cost-effective"
        ],
        max_steps=25
    )
    
    print("\n\nARCHITECTURE DESIGN:")
    print("=" * 50)
    for step in steps:
        print(f"\n{step.step_number}. {step.action}")
        print(f"   Rationale: {step.reasoning}")
    
    thinking.close()


async def use_for_problem_solving():
    """Example: Solve a complex optimization problem."""
    thinking = get_mac_optimized_thinking()
    
    steps = await thinking.think(
        goal="Optimize database query that takes 30 seconds",
        constraints=[
            "Cannot change schema",
            "Must maintain ACID properties",
            "Limited to 16GB RAM",
            "Need sub-second response time"
        ],
        max_steps=15
    )
    
    print("\n\nOPTIMIZATION SOLUTION:")
    print("=" * 50)
    for step in steps:
        print(f"\n{step.step_number}. {step.action}")
        print(f"   Implementation: {step.reasoning}")
    
    thinking.close()


async def use_for_debugging():
    """Example: Debug a complex issue."""
    thinking = get_mac_optimized_thinking()
    
    steps = await thinking.think(
        goal="Debug intermittent memory leak in production service",
        constraints=[
            "Cannot reproduce locally",
            "Happens once per day",
            "No obvious pattern",
            "Need minimal performance impact"
        ],
        max_steps=20
    )
    
    print("\n\nDEBUGGING APPROACH:")
    print("=" * 50)
    for step in steps:
        print(f"\n{step.step_number}. {step.action}")
    
    thinking.close()


async def use_with_custom_context():
    """Example: Use with rich context."""
    thinking = get_mac_optimized_thinking()
    
    # You can provide context about your specific situation
    context = {
        "current_tech_stack": ["Python", "PostgreSQL", "Redis", "Kubernetes"],
        "team_size": 5,
        "timeline": "3 months",
        "budget": "$50K",
        "existing_issues": ["High latency", "Complex deployment", "No monitoring"]
    }
    
    steps = await thinking.think(
        goal="Modernize our API infrastructure",
        constraints=[
            "Zero downtime migration",
            "Reuse existing team skills",
            "Stay within budget",
            "Improve observability"
        ],
        initial_state=context,
        max_steps=30
    )
    
    print("\n\nMODERNIZATION PLAN:")
    print("=" * 50)
    
    # Group by confidence level
    high_confidence = [s for s in steps if s.confidence > 0.8]
    medium_confidence = [s for s in steps if 0.5 <= s.confidence <= 0.8]
    
    print("\nHigh Confidence Steps:")
    for step in high_confidence[:5]:
        print(f"  • {step.action}")
    
    print("\nMedium Confidence Steps:")
    for step in medium_confidence[:5]:
        print(f"  • {step.action}")
    
    thinking.close()


def create_thinking_wrapper():
    """Create a reusable wrapper for your specific domain."""
    
    class TradingThinking:
        def __init__(self):
            self.thinking = get_mac_optimized_thinking()
            
        async def optimize_strategy(self, current_strategy: dict) -> list:
            """Optimize a trading strategy."""
            steps = await self.thinking.think(
                goal=f"Optimize {current_strategy['name']} strategy",
                constraints=[
                    "Maintain risk limits",
                    "Reduce drawdown by 20%",
                    "Improve Sharpe ratio",
                    "Keep execution simple"
                ],
                initial_state=current_strategy,
                max_steps=20
            )
            return steps
            
        async def analyze_market_condition(self, market_data: dict) -> list:
            """Analyze market conditions."""
            steps = await self.thinking.think(
                goal="Identify optimal trading opportunities",
                constraints=[
                    "Consider current volatility",
                    "Account for correlation changes",
                    "Respect position limits",
                    "Minimize market impact"
                ],
                initial_state=market_data,
                max_steps=15
            )
            return steps
            
        def close(self):
            self.thinking.close()
    
    return TradingThinking()


async def practical_example():
    """A practical example you can adapt."""
    
    # Initialize
    thinking = get_mac_optimized_thinking()
    
    # Define your problem
    my_problem = "Reduce AWS costs by 40% without impacting performance"
    
    my_constraints = [
        "Cannot change core architecture",
        "Must maintain 99.9% uptime",
        "Keep response time under 200ms",
        "No data loss",
        "Complete in 30 days"
    ]
    
    # Add context about your current situation
    my_context = {
        "current_monthly_cost": 50000,
        "main_services": ["EC2", "RDS", "S3", "CloudFront"],
        "traffic_pattern": "Peaks at 9am-5pm EST",
        "data_size": "500TB",
        "team_expertise": ["Python", "Terraform", "AWS"]
    }
    
    # Get solution
    print("ANALYZING YOUR PROBLEM...")
    print("=" * 50)
    
    steps = await thinking.think(
        goal=my_problem,
        constraints=my_constraints,
        initial_state=my_context,
        max_steps=25
    )
    
    # Present results in actionable format
    print(f"\nGENERATED {len(steps)} OPTIMIZATION STEPS:")
    print("=" * 50)
    
    # Prioritize by confidence
    steps_by_priority = sorted(steps, key=lambda s: s.confidence, reverse=True)
    
    print("\n🎯 TOP PRIORITY ACTIONS (High Confidence):")
    for i, step in enumerate(steps_by_priority[:5], 1):
        print(f"\n{i}. {step.action}")
        print(f"   Impact: {step.reasoning}")
        print(f"   Confidence: {step.confidence:.0%}")
    
    print("\n📋 ADDITIONAL RECOMMENDATIONS:")
    for step in steps_by_priority[5:10]:
        print(f"  • {step.action} ({step.confidence:.0%})")
    
    # Save results
    with open("cost_optimization_plan.txt", "w") as f:
        f.write(f"AWS Cost Optimization Plan\n")
        f.write("=" * 50 + "\n\n")
        for step in steps:
            f.write(f"{step.step_number}. {step.action}\n")
            f.write(f"   {step.reasoning}\n")
            f.write(f"   Confidence: {step.confidence:.0%}\n\n")
    
    print("\n✅ Plan saved to cost_optimization_plan.txt")
    
    thinking.close()


async def main():
    """Run examples."""
    
    # Choose which example to run
    print("Hardware-Accelerated Sequential Thinking - Real Usage Examples")
    print("=" * 60)
    
    # Run practical example
    await practical_example()
    
    # Uncomment to run other examples:
    # await use_for_coding_task()
    # await use_for_architecture_design()
    # await use_for_problem_solving()
    # await use_for_debugging()
    # await use_with_custom_context()


if __name__ == "__main__":
    asyncio.run(main())