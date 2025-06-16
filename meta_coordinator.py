"""
MetaCoordinator - The Meta System Orchestrator

This coordinates the entire meta system and provides the evolutionary intelligence.
It watches the watchers, evolves the evolvers, and bootstraps new capabilities.

Design Decision: Central coordination vs distributed autonomous agents
Rationale: Central coordinator provides coherent evolution direction and prevents conflicts
Alternative: Fully distributed meta-agents
Prediction: Will enable more sophisticated and coordinated evolution patterns
"""

import asyncio
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from meta_auditor import MetaAuditor
from meta_config import get_meta_config
from meta_executor import MetaExecutor
from meta_generator import MetaGenerator
from meta_prime import MetaPrime
from meta_watcher import MetaWatcher


@dataclass
class EvolutionPlan:
    """Plan for evolutionary changes"""

    plan_id: str
    target_component: str  # Which component to evolve
    evolution_type: str  # Type of evolution (capability_addition, optimization, restructure)
    rationale: str  # Why this evolution is needed
    implementation: str  # How to implement it
    confidence: float  # Confidence in the plan
    expected_benefit: str  # Expected improvement
    risk_assessment: str  # Potential risks
    dependencies: list[str]  # Other components that need to exist first


class MetaCoordinator:
    """Orchestrates the meta system and drives evolution"""

    def __init__(self):
        self.birth_time = time.time()
        self.config = get_meta_config()
        self.db = sqlite3.connect(self.config.database.evolution_db)

        # Initialize components
        # Only create MetaPrime if not disabled
        import os

        if os.environ.get("DISABLE_META_AUTOSTART") == "1":
            self.meta_prime = None
        else:
            self.meta_prime = MetaPrime()
        self.meta_watcher = MetaWatcher()
        self.meta_generator = MetaGenerator()
        self.meta_auditor = MetaAuditor()
        self.meta_executor = MetaExecutor()

        # Evolution state
        self.evolution_plans: list[EvolutionPlan] = []
        self.generation_count = 0
        self.capabilities = set(
            [
                "observe",
                "record",
                "analyze",
                "generate_code",
                "self_modify",
                "self_audit",
                "execute_real_changes",
            ]
        )

        # Pattern detection
        self.known_patterns = {}
        self.evolution_triggers = {}

        print(f"🧠 MetaCoordinator initialized at {time.ctime(self.birth_time)}")
        print(f"📊 Current capabilities: {self.capabilities}")

        self._init_coordinator_schema()
        self._record_birth()

        # Record any evolutionary learnings from previous runs
        self._record_evolution_learning(
            "recursion_bug_fix",
            "Fixed circular dependency in analyze_system_state -> assess_evolution_readiness",
            "Added include_evolution_readiness parameter to break recursion loop",
        )

    def _init_coordinator_schema(self):
        """Initialize coordinator-specific database schema"""

        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS evolution_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                plan_id TEXT UNIQUE NOT NULL,
                target_component TEXT NOT NULL,
                evolution_type TEXT NOT NULL,
                rationale TEXT NOT NULL,
                implementation TEXT NOT NULL,
                confidence REAL NOT NULL,
                expected_benefit TEXT NOT NULL,
                risk_assessment TEXT NOT NULL,
                status TEXT DEFAULT 'planned',
                execution_timestamp REAL,
                outcome TEXT
            )
        """
        )

        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS coordination_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                source_component TEXT NOT NULL,
                target_component TEXT,
                details TEXT NOT NULL,
                impact_assessment TEXT
            )
        """
        )

        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS capability_evolution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                capability_name TEXT NOT NULL,
                evolution_generation INTEGER NOT NULL,
                implementation_code TEXT,
                performance_metrics TEXT,
                success_indicators TEXT
            )
        """
        )

        self.db.commit()

    def _record_birth(self):
        """Record the birth of the coordinator"""

        self.db.execute(
            """
            INSERT INTO coordination_events 
            (timestamp, event_type, source_component, details, impact_assessment)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                time.time(),
                "coordinator_birth",
                "MetaCoordinator",
                "Central coordination system initialized",
                "Enables coordinated evolution of meta system",
            ),
        )

        self.db.commit()

    def _record_evolution_learning(
        self, learning_type: str, description: str, implementation: str
    ):
        """Record what the system learned from evolution"""

        self.db.execute(
            """
            INSERT INTO coordination_events 
            (timestamp, event_type, source_component, details, impact_assessment)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                time.time(),
                "evolution_learning",
                "MetaCoordinator",
                f"{learning_type}: {description}",
                f"Implementation: {implementation}",
            ),
        )

        self.db.commit()
        print(f"📚 Evolutionary Learning Recorded: {learning_type}")

    def record_coordination_event(
        self,
        event_type: str,
        source_component: str,
        details: str,
        impact: str,
        target_component: str = None,
    ):
        """Record coordination events between meta components"""

        self.db.execute(
            """
            INSERT INTO coordination_events 
            (timestamp, event_type, source_component, target_component, details, impact_assessment)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                time.time(),
                event_type,
                source_component,
                target_component,
                details,
                impact,
            ),
        )

        self.db.commit()
        print(f"🔄 Coordination Event: {event_type} from {source_component}")

    def analyze_system_state(
        self, include_evolution_readiness: bool = True
    ) -> dict[str, Any]:
        """Analyze the current state of the entire meta system"""

        # Get observations from all components
        cursor = self.db.execute(
            """
            SELECT event_type, COUNT(*) as count, MAX(timestamp) as latest
            FROM observations 
            GROUP BY event_type
            ORDER BY count DESC
        """
        )

        event_summary = {
            row[0]: {"count": row[1], "latest": row[2]} for row in cursor.fetchall()
        }

        # Analyze recent activity
        recent_threshold = (
            time.time() - self.config.timing.recent_activity_window_seconds
        )
        cursor = self.db.execute(
            """
            SELECT COUNT(*) FROM observations WHERE timestamp > ?
        """,
            (recent_threshold,),
        )

        recent_activity = cursor.fetchone()[0]

        # Check for patterns
        patterns = self.detect_meta_patterns()

        # Assess evolution readiness (but avoid recursion)
        evolution_readiness = None
        if include_evolution_readiness:
            evolution_readiness = self.assess_evolution_readiness()

        result = {
            "system_age_minutes": (time.time() - self.birth_time) / 60,
            "total_events": sum(e["count"] for e in event_summary.values()),
            "recent_activity": recent_activity,
            "event_types": len(event_summary),
            "patterns_detected": len(patterns),
            "current_capabilities": list(self.capabilities),
            "generation": self.generation_count,
        }

        if evolution_readiness is not None:
            result["evolution_readiness"] = evolution_readiness

        return result

    def detect_meta_patterns(self) -> dict[str, Any]:
        """Detect high-level patterns across the meta system"""

        patterns = {}

        # Pattern 1: Rapid development cycles
        cursor = self.db.execute(
            """
            SELECT timestamp FROM observations 
            WHERE event_type = 'file_modified'
            ORDER BY timestamp DESC LIMIT 10
        """
        )

        mod_times = [row[0] for row in cursor.fetchall()]
        if len(mod_times) >= 3:
            intervals = [
                mod_times[i] - mod_times[i + 1] for i in range(len(mod_times) - 1)
            ]
            avg_interval = sum(intervals) / len(intervals)

            if avg_interval < self.config.timing.rapid_development_threshold_seconds:
                patterns["rapid_development"] = {
                    "detected": True,
                    "average_interval": avg_interval,
                    "interpretation": "Active meta-system construction",
                }

        # Pattern 2: Component interaction frequency
        cursor = self.db.execute(
            """
            SELECT context, COUNT(*) as count
            FROM observations 
            WHERE context IS NOT NULL
            GROUP BY context
        """
        )

        component_activity = {row[0]: row[1] for row in cursor.fetchall()}
        if len(component_activity) > 1:
            patterns["component_interaction"] = {
                "active_components": len(component_activity),
                "activity_distribution": component_activity,
            }

        # Pattern 3: Evolution triggers
        cursor = self.db.execute(
            """
            SELECT event_type, COUNT(*) as count
            FROM observations 
            WHERE event_type LIKE '%pattern%' OR event_type LIKE '%evolution%'
            GROUP BY event_type
        """
        )

        evolution_signals = {row[0]: row[1] for row in cursor.fetchall()}
        if evolution_signals:
            patterns["evolution_signals"] = evolution_signals

        return patterns

    def assess_evolution_readiness(self) -> dict[str, Any]:
        """Assess if the system is ready for evolutionary changes"""

        state = self.analyze_system_state(include_evolution_readiness=False)
        patterns = self.detect_meta_patterns()

        # Readiness criteria
        sufficient_observations = (
            state["total_events"]
            >= self.config.timing.minimum_observations_for_evolution
        )
        recent_activity = (
            state["recent_activity"] >= self.config.timing.minimum_recent_activity_count
        )
        pattern_detected = len(patterns) >= 1
        stable_operation = state["system_age_minutes"] >= 1

        readiness_score = sum(
            [
                sufficient_observations * 0.3,
                recent_activity * 0.3,
                pattern_detected * 0.2,
                stable_operation * 0.2,
            ]
        )

        return {
            "ready": readiness_score >= 0.7,
            "score": readiness_score,
            "criteria": {
                "sufficient_observations": sufficient_observations,
                "recent_activity": recent_activity,
                "pattern_detected": pattern_detected,
                "stable_operation": stable_operation,
            },
            "next_evolution_type": self._suggest_next_evolution(patterns),
        }

    def _suggest_next_evolution(self, patterns: dict[str, Any]) -> str:
        """Suggest what type of evolution should happen next"""

        current_caps = len(self.capabilities)

        if "rapid_development" in patterns:
            if current_caps < 5:
                return "capability_addition"
            else:
                return "optimization"
        elif "component_interaction" in patterns:
            return "integration_improvement"
        else:
            return "observational_enhancement"

    def generate_evolution_plan(self) -> EvolutionPlan | None:
        """Generate a concrete evolution plan"""

        readiness = self.assess_evolution_readiness()

        if not readiness["ready"]:
            return None

        evolution_type = readiness["next_evolution_type"]
        plan_id = f"evo_{self.generation_count}_{int(time.time())}"

        if evolution_type == "capability_addition":
            return self._plan_capability_addition(plan_id)
        elif evolution_type == "optimization":
            return self._plan_optimization(plan_id)
        elif evolution_type == "integration_improvement":
            return self._plan_integration_improvement(plan_id)
        else:
            return self._plan_observational_enhancement(plan_id)

    def _plan_capability_addition(self, plan_id: str) -> EvolutionPlan:
        """Plan adding a new capability"""

        # Analyze what capability is most needed
        state = self.analyze_system_state()

        if "code_generation" not in self.capabilities:
            target_capability = "code_generation"
            implementation = """
            Add CodeGenerator class that can:
            1. Analyze existing code patterns
            2. Generate new code based on observations
            3. Validate generated code for syntax
            4. Track generation success rates
            """
        elif "pattern_learning" not in self.capabilities:
            target_capability = "pattern_learning"
            implementation = """
            Add PatternLearner class that can:
            1. Extract recurring patterns from observations
            2. Build prediction models
            3. Suggest optimizations based on patterns
            4. Learn from evolution outcomes
            """
        else:
            target_capability = "self_modification"
            implementation = """
            Add SelfModifier class that can:
            1. Modify its own source code
            2. Create backup before modifications
            3. Test modifications safely
            4. Rollback if modifications fail
            """

        return EvolutionPlan(
            plan_id=plan_id,
            target_component="MetaCoordinator",
            evolution_type="capability_addition",
            rationale=f"System shows readiness for {target_capability} based on {state['recent_activity']} recent activities",
            implementation=implementation,
            confidence=0.8,
            expected_benefit=f"Adds {target_capability} to enable next level of meta-programming",
            risk_assessment="Low risk - additive change, existing functionality preserved",
            dependencies=["meta_prime", "meta_watcher"],
        )

    def _plan_optimization(self, plan_id: str) -> EvolutionPlan:
        """Plan optimization improvements"""

        return EvolutionPlan(
            plan_id=plan_id,
            target_component="entire_system",
            evolution_type="optimization",
            rationale="System has sufficient capabilities, time to optimize performance",
            implementation="Profile all operations, optimize slow paths, add caching",
            confidence=0.7,
            expected_benefit="Faster observation processing and pattern detection",
            risk_assessment="Medium risk - changing existing code paths",
            dependencies=["all_components"],
        )

    def _plan_integration_improvement(self, plan_id: str) -> EvolutionPlan:
        """Plan better integration between components"""

        return EvolutionPlan(
            plan_id=plan_id,
            target_component="component_interfaces",
            evolution_type="integration_improvement",
            rationale="Multiple components active, need better coordination",
            implementation="Add event bus, standardize interfaces, improve data sharing",
            confidence=0.6,
            expected_benefit="Better coordination and information flow between components",
            risk_assessment="Medium risk - affects multiple components",
            dependencies=["meta_prime", "meta_watcher", "meta_coordinator"],
        )

    def _plan_observational_enhancement(self, plan_id: str) -> EvolutionPlan:
        """Plan improvements to observation capabilities"""

        return EvolutionPlan(
            plan_id=plan_id,
            target_component="meta_watcher",
            evolution_type="observational_enhancement",
            rationale="Need deeper insights into system behavior",
            implementation="Add code change analysis, performance monitoring, behavior prediction",
            confidence=0.9,
            expected_benefit="Better understanding of development patterns and bottlenecks",
            risk_assessment="Low risk - enhances existing observation without breaking changes",
            dependencies=["meta_watcher"],
        )

    def execute_evolution_plan(self, plan: EvolutionPlan) -> bool:
        """Execute an evolution plan using the code generator"""

        # Record the plan
        self.db.execute(
            """
            INSERT INTO evolution_plans 
            (timestamp, plan_id, target_component, evolution_type, rationale, 
             implementation, confidence, expected_benefit, risk_assessment, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                time.time(),
                plan.plan_id,
                plan.target_component,
                plan.evolution_type,
                plan.rationale,
                plan.implementation,
                plan.confidence,
                plan.expected_benefit,
                plan.risk_assessment,
                "executing",
            ),
        )

        self.db.commit()

        print(f"🧬 Executing Evolution Plan: {plan.plan_id}")
        print(f"   Type: {plan.evolution_type}")
        print(f"   Target: {plan.target_component}")
        print(f"   Confidence: {plan.confidence:.1%}")

        success = False

        if plan.evolution_type == "capability_addition":
            # Use generator to create new capabilities AND actually apply them
            generations = self.meta_generator.plan_code_generation()

            for generation in generations:
                # First simulate to check viability
                result = self.meta_generator.simulate_code_application(generation)

                if result["would_succeed"] and generation.confidence > 0.7:
                    print(f"   🚀 Executing real change: {generation.purpose}")

                    # Actually execute the code generation
                    executed = self.meta_executor.execute_code_generation(generation)
                    if executed:
                        print(f"   ✅ Real code change applied: {generation.purpose}")
                        success = True
                    else:
                        print(f"   ❌ Execution failed: {generation.purpose}")
                elif result["would_succeed"]:
                    print(f"   🔮 Low confidence, simulated only: {generation.purpose}")
                    success = True

            # Add capabilities (simulated)
            if "code_generation" in plan.implementation:
                self.capabilities.add("advanced_code_generation")
            elif "pattern_learning" in plan.implementation:
                self.capabilities.add("pattern_learning")
            elif "self_modification" in plan.implementation:
                self.capabilities.add("self_modification")

        elif plan.evolution_type == "optimization":
            print(f"   🚀 Would optimize: {plan.target_component}")
            success = True

        elif plan.evolution_type == "integration_improvement":
            print(f"   🔗 Would improve integration: {plan.target_component}")
            success = True

        else:
            print(f"   📈 Would enhance: {plan.target_component}")
            success = True

        # Update plan status
        status = "completed" if success else "failed"
        self.db.execute(
            """
            UPDATE evolution_plans 
            SET status = ?, execution_timestamp = ?, outcome = ?
            WHERE plan_id = ?
        """,
            (
                status,
                time.time(),
                "simulated_success" if success else "failed",
                plan.plan_id,
            ),
        )

        self.db.commit()

        if success:
            self.generation_count += 1
            print(f"   🎉 Evolution successful! Generation: {self.generation_count}")

        return success

    async def coordination_loop(self):
        """Main coordination loop"""

        print("🚀 Starting meta coordination loop...")

        loop_count = 0

        while True:
            loop_count += 1

            # Analyze current state
            state = self.analyze_system_state()

            # Check for evolution opportunities
            evolution_plan = self.generate_evolution_plan()

            if evolution_plan:
                success = self.execute_evolution_plan(evolution_plan)
                if success:
                    print(f"✨ Evolution executed: {evolution_plan.evolution_type}")

            # Record coordination cycle
            self.db.execute(
                """
                INSERT INTO coordination_events 
                (timestamp, event_type, source_component, details, impact_assessment)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    time.time(),
                    "coordination_cycle",
                    "MetaCoordinator",
                    f"Loop {loop_count}: {state['recent_activity']} recent events",
                    f"System generation {self.generation_count}",
                ),
            )
            self.db.commit()

            # Status report and self-audit every 10 cycles
            if loop_count % 10 == 0:
                print(f"\n📊 Coordination Status (Loop {loop_count}):")
                print(f"   System age: {state['system_age_minutes']:.1f} minutes")
                print(f"   Total events: {state['total_events']}")
                print(f"   Recent activity: {state['recent_activity']}")
                print(f"   Generation: {self.generation_count}")
                print(f"   Capabilities: {len(self.capabilities)}")

                # Self-audit every 20 cycles using Claude's patterns
                if loop_count % 20 == 0:
                    print(
                        f"\n🔍 Running self-audit (Generation {self.generation_count})..."
                    )
                    audit_summary = self.meta_auditor.audit_meta_system(
                        self.generation_count
                    )
                    print(f"   Audit Status: {audit_summary['overall_status']}")
                    if audit_summary["top_concerns"]:
                        print(
                            f"   Priority Fixes: {len(audit_summary['top_concerns'])}"
                        )

                    # Learn from audit results
                    if audit_summary["overall_status"] == "NEEDS_WORK":
                        self._record_evolution_learning(
                            "self_audit_findings",
                            f"Found {audit_summary['needs_work_count']} issues in generation {self.generation_count}",
                            f"Top concerns: {', '.join(audit_summary['top_concerns'][:3])}",
                        )

            # Coordinate at configured intervals
            await asyncio.sleep(self.config.timing.coordination_cycle_seconds)

    def get_coordination_report(self) -> str:
        """Generate a comprehensive coordination report"""

        state = self.analyze_system_state()
        patterns = self.detect_meta_patterns()
        readiness = self.assess_evolution_readiness()

        report = f"""
🧠 Meta Coordination Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

System Overview:
  Age: {state['system_age_minutes']:.1f} minutes
  Generation: {self.generation_count}
  Total Events: {state['total_events']}
  Recent Activity: {state['recent_activity']}
  
Capabilities: {', '.join(sorted(self.capabilities))}

Evolution Status:
  Ready: {'🟢 Yes' if readiness['ready'] else '🟡 Not yet'}
  Readiness Score: {readiness['score']:.1%}
  Next Evolution: {readiness['next_evolution_type']}

Patterns Detected: {len(patterns)}
{chr(10).join(f"  • {name}: {info}" for name, info in patterns.items())}

Latest Evolution Plans:
"""

        # Get recent evolution plans
        cursor = self.db.execute(
            """
            SELECT plan_id, evolution_type, target_component, confidence, status
            FROM evolution_plans 
            ORDER BY timestamp DESC LIMIT 3
        """
        )

        for plan_id, evo_type, target, confidence, status in cursor.fetchall():
            report += f"  • {plan_id}: {evo_type} on {target} ({confidence:.1%} confidence, {status})\n"

        return report


async def start_meta_coordination():
    """Start the complete meta system with coordination"""

    coordinator = MetaCoordinator()

    print(coordinator.get_coordination_report())

    # Start file watching in background
    watcher_task = asyncio.create_task(coordinator.meta_watcher.watch_files())

    # Add files to watch
    coordinator.meta_watcher.add_file_to_watch(Path("meta_prime.py"))
    coordinator.meta_watcher.add_file_to_watch(Path("meta_watcher.py"))
    coordinator.meta_watcher.add_file_to_watch(Path("meta_coordinator.py"))
    coordinator.meta_watcher.add_file_to_watch(Path("meta_generator.py"))
    coordinator.meta_watcher.add_file_to_watch(Path("meta_auditor.py"))
    coordinator.meta_watcher.add_file_to_watch(Path("meta_executor.py"))

    # Start coordination
    try:
        await coordinator.coordination_loop()
    except KeyboardInterrupt:
        print("\n🛑 Meta coordination stopped")
        print(coordinator.get_coordination_report())
        watcher_task.cancel()


if __name__ == "__main__":
    import argparse
    import os

    # Check if meta system is disabled
    if os.environ.get("DISABLE_META_AUTOSTART") == "1":
        print("⚠️ Meta Coordinator blocked by DISABLE_META_AUTOSTART=1")
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Meta Coordination System")
    parser.add_argument(
        "--dev-mode", action="store_true", help="Start in development observation mode"
    )
    parser.add_argument(
        "--build-mode", action="store_true", help="Start in build coordination mode"
    )
    parser.add_argument(
        "--test-mode", action="store_true", help="Start in test observation mode"
    )
    parser.add_argument("--status", action="store_true", help="Show meta system status")
    parser.add_argument(
        "--evolve-now", action="store_true", help="Trigger immediate evolution"
    )

    args = parser.parse_args()

    if args.status:
        coordinator = MetaCoordinator()
        print(coordinator.get_coordination_report())
    elif args.evolve_now:
        coordinator = MetaCoordinator()
        plan = coordinator.generate_evolution_plan()
        if plan:
            coordinator.execute_evolution_plan(plan)
    elif args.dev_mode:
        print("🔧 Starting Meta System in Development Mode...")
        coordinator = MetaCoordinator()
        coordinator.record_coordination_event(
            "dev_mode_start",
            "MetaCoordinator",
            "Development observation mode activated",
            "Enhanced code observation",
        )
        asyncio.run(start_meta_coordination())
    elif args.build_mode:
        print("🏗️ Starting Meta System in Build Mode...")
        coordinator = MetaCoordinator()
        coordinator.record_coordination_event(
            "build_mode_start",
            "MetaCoordinator",
            "Build coordination mode activated",
            "Enhanced build process monitoring",
        )
        # Focus on build-related observations
        print("Build mode active - monitoring for build patterns and issues")
        print(coordinator.get_coordination_report())
    elif args.test_mode:
        print("🧪 Starting Meta System in Test Mode...")
        coordinator = MetaCoordinator()
        coordinator.record_coordination_event(
            "test_mode_start",
            "MetaCoordinator",
            "Test observation mode activated",
            "Enhanced test pattern learning",
        )
        # Focus on test-related observations
        print("Test mode active - monitoring test patterns and outcomes")
        print(coordinator.get_coordination_report())
    else:
        # Default: full coordination mode
        asyncio.run(start_meta_coordination())
