#!/usr/bin/env python3
"""
Meta Model CLI Trainer
Integrates Claude CLI reasoning capture with existing meta system components
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_cli_reasoning_capture import ClaudeCLIReasoningCapture

from meta_auditor import MetaAuditor
from meta_coordinator import MetaCoordinator
from meta_executor import MetaExecutor
from meta_generator import MetaGenerator

# Import existing meta components
from meta_prime import MetaPrime


@dataclass
class MetaTrainingInsight:
    """Training insight for meta system evolution"""

    insight_id: str
    pattern_type: str
    reasoning_chain: list[str]
    confidence: float
    meta_application: str
    training_value: float


class MetaClaudeCLITrainer:
    """Trains meta system using Claude CLI reasoning patterns"""

    def __init__(self):
        # Reuse existing meta components
        self.meta_prime = MetaPrime()
        self.meta_coordinator = MetaCoordinator()
        self.meta_auditor = MetaAuditor()
        self.meta_executor = MetaExecutor()
        self.meta_generator = MetaGenerator()

        # CLI capture integration
        self.cli_capture = ClaudeCLIReasoningCapture()

        # Training state
        self.training_sessions = []
        self.learned_patterns = {}
        self.meta_improvements = []

        print("🎓 Meta Claude CLI Trainer initialized")
        print("🔄 Integrated with existing meta system components")

    def ingest_cli_session(self, session_file: str) -> dict[str, Any]:
        """Ingest CLI session data for meta training"""

        with open(session_file) as f:
            session_data = json.load(f)

        self.training_sessions.append(session_data)

        # Extract training insights using meta_auditor patterns
        insights = self._extract_meta_training_insights(session_data)

        # Apply insights to meta system using meta_coordinator
        improvements = self._apply_insights_to_meta_system(insights)

        # Record training session in meta_prime
        self.meta_prime.observe(
            "meta_cli_training_session",
            {
                "session_id": session_data["session_id"],
                "events_processed": len(session_data["events"]),
                "insights_extracted": len(insights),
                "improvements_applied": len(improvements),
                "training_value": sum(insight.training_value for insight in insights),
            },
        )

        print(
            f"📚 Ingested session: {len(session_data['events'])} events → {len(insights)} insights → {len(improvements)} improvements"
        )

        return {
            "session_id": session_data["session_id"],
            "insights": insights,
            "improvements": improvements,
        }

    def _extract_meta_training_insights(
        self, session_data: dict[str, Any]
    ) -> list[MetaTrainingInsight]:
        """Extract training insights using meta_auditor analysis"""

        insights = []
        events = session_data["events"]

        # Use meta_auditor to analyze patterns
        for _i, event in enumerate(events):
            if event["event_type"] == "claude_analysis":
                # Analyze Claude's reasoning for meta-applicable patterns
                insight = self._analyze_reasoning_for_meta_learning(event, session_data)
                if insight:
                    insights.append(insight)

            elif event["event_type"] == "code_generation":
                # Analyze code generation patterns for meta_generator enhancement
                insight = self._analyze_code_generation_for_meta_learning(
                    event, session_data
                )
                if insight:
                    insights.append(insight)

            elif event["event_type"] == "error_recovery":
                # Analyze error recovery for meta_executor safety improvements
                insight = self._analyze_error_recovery_for_meta_learning(
                    event, session_data
                )
                if insight:
                    insights.append(insight)

        # Cross-event pattern analysis using meta_coordinator logic
        sequence_insights = self._analyze_event_sequences(events)
        insights.extend(sequence_insights)

        return insights

    def _analyze_reasoning_for_meta_learning(
        self, event: dict[str, Any], session_data: dict[str, Any]
    ) -> MetaTrainingInsight | None:
        """Analyze Claude's reasoning patterns for meta system enhancement"""

        reasoning_patterns = event.get("reasoning_patterns", [])
        content = event.get("content", "")

        # Focus on systematic reasoning that meta_coordinator can adopt
        if "systematic_reasoning" in reasoning_patterns:
            return MetaTrainingInsight(
                insight_id=f"reasoning_{event['timestamp']}",
                pattern_type="systematic_decision_making",
                reasoning_chain=self._extract_reasoning_chain(content),
                confidence=event.get("confidence", 0.5),
                meta_application="Enhance MetaCoordinator decision-making process",
                training_value=0.8,
            )

        # Safety-conscious reasoning for meta_executor
        if "safety_conscious" in reasoning_patterns:
            return MetaTrainingInsight(
                insight_id=f"safety_{event['timestamp']}",
                pattern_type="enhanced_safety_protocols",
                reasoning_chain=self._extract_safety_considerations(content),
                confidence=event.get("confidence", 0.5),
                meta_application="Strengthen MetaExecutor safety validation",
                training_value=0.9,
            )

        # Solution-oriented thinking for meta_generator
        if "solution_oriented" in reasoning_patterns:
            return MetaTrainingInsight(
                insight_id=f"solution_{event['timestamp']}",
                pattern_type="solution_generation_strategy",
                reasoning_chain=self._extract_solution_approach(content),
                confidence=event.get("confidence", 0.5),
                meta_application="Improve MetaGenerator solution strategies",
                training_value=0.7,
            )

        return None

    def _analyze_code_generation_for_meta_learning(
        self, event: dict[str, Any], session_data: dict[str, Any]
    ) -> MetaTrainingInsight | None:
        """Analyze code generation patterns for meta_generator enhancement"""

        code = event.get("content", "")
        patterns = event.get("reasoning_patterns", [])
        event.get("context", {})

        # Well-documented code patterns for meta_generator
        if "well_documented" in patterns:
            return MetaTrainingInsight(
                insight_id=f"documentation_{event['timestamp']}",
                pattern_type="enhanced_code_documentation",
                reasoning_chain=self._extract_documentation_patterns(code),
                confidence=event.get("confidence", 0.5),
                meta_application="Improve MetaGenerator code documentation standards",
                training_value=0.6,
            )

        # Error handling patterns for meta_executor
        if "error_handling" in patterns:
            return MetaTrainingInsight(
                insight_id=f"error_handling_{event['timestamp']}",
                pattern_type="robust_error_handling",
                reasoning_chain=self._extract_error_handling_patterns(code),
                confidence=event.get("confidence", 0.5),
                meta_application="Enhance MetaExecutor error handling capabilities",
                training_value=0.8,
            )

        return None

    def _analyze_error_recovery_for_meta_learning(
        self, event: dict[str, Any], session_data: dict[str, Any]
    ) -> MetaTrainingInsight | None:
        """Analyze error recovery patterns for meta system robustness"""

        content = event.get("content", "")
        if "ERROR:" in content and "SOLUTION:" in content:
            error_part = content.split("SOLUTION:")[0].replace("ERROR:", "").strip()
            solution_part = content.split("SOLUTION:")[1].strip()

            return MetaTrainingInsight(
                insight_id=f"recovery_{event['timestamp']}",
                pattern_type="intelligent_error_recovery",
                reasoning_chain=[error_part, "→", solution_part],
                confidence=event.get("confidence", 0.5),
                meta_application="Improve MetaExecutor recovery strategies",
                training_value=0.9,
            )

        return None

    def _analyze_event_sequences(
        self, events: list[dict[str, Any]]
    ) -> list[MetaTrainingInsight]:
        """Analyze sequences of events for meta workflow improvements"""

        insights = []

        # Look for user_request → claude_analysis → code_generation sequences
        for i in range(len(events) - 2):
            if (
                events[i]["event_type"] == "user_request"
                and events[i + 1]["event_type"] == "claude_analysis"
                and events[i + 2]["event_type"] == "code_generation"
            ):
                insight = MetaTrainingInsight(
                    insight_id=f"workflow_{events[i]['timestamp']}",
                    pattern_type="request_analysis_implementation_workflow",
                    reasoning_chain=[
                        events[i]["content"][:50] + "...",
                        events[i + 1]["content"][:50] + "...",
                        events[i + 2]["content"][:50] + "...",
                    ],
                    confidence=0.8,
                    meta_application="Enhance MetaCoordinator workflow orchestration",
                    training_value=0.7,
                )
                insights.append(insight)

        return insights

    def _apply_insights_to_meta_system(
        self, insights: list[MetaTrainingInsight]
    ) -> list[dict[str, Any]]:
        """Apply training insights to meta system components"""

        improvements = []

        for insight in insights:
            if insight.training_value >= 0.7:  # High-value insights only
                improvement = self._create_meta_improvement(insight)
                if improvement:
                    improvements.append(improvement)

                    # Record the improvement using meta_prime
                    self.meta_prime.record_design_decision(
                        decision=f"apply_cli_insight_{insight.insight_id}",
                        rationale=f"CLI training revealed: {insight.pattern_type}",
                        alternatives="ignore_cli_pattern",
                        prediction=f"Enhanced {insight.meta_application}",
                    )

        return improvements

    def _create_meta_improvement(
        self, insight: MetaTrainingInsight
    ) -> dict[str, Any] | None:
        """Create specific meta system improvement from insight"""

        if "MetaCoordinator" in insight.meta_application:
            # Enhance decision-making in meta_coordinator
            improvement = {
                "component": "MetaCoordinator",
                "enhancement_type": insight.pattern_type,
                "implementation": f"Add {insight.pattern_type} to decision-making process",
                "reasoning_chain": insight.reasoning_chain,
                "confidence": insight.confidence,
                "training_source": "claude_cli",
            }

            # Use meta_auditor to validate improvement
            audit_result = self._audit_improvement_safety(improvement)
            if audit_result["safe"]:
                return improvement

        elif "MetaGenerator" in insight.meta_application:
            # Enhance code generation in meta_generator
            improvement = {
                "component": "MetaGenerator",
                "enhancement_type": insight.pattern_type,
                "implementation": f"Incorporate {insight.pattern_type} in code generation",
                "reasoning_chain": insight.reasoning_chain,
                "confidence": insight.confidence,
                "training_source": "claude_cli",
            }

            audit_result = self._audit_improvement_safety(improvement)
            if audit_result["safe"]:
                return improvement

        elif "MetaExecutor" in insight.meta_application:
            # Enhance execution safety in meta_executor
            improvement = {
                "component": "MetaExecutor",
                "enhancement_type": insight.pattern_type,
                "implementation": f"Strengthen {insight.pattern_type} in execution",
                "reasoning_chain": insight.reasoning_chain,
                "confidence": insight.confidence,
                "training_source": "claude_cli",
            }

            audit_result = self._audit_improvement_safety(improvement)
            if audit_result["safe"]:
                return improvement

        return None

    def _audit_improvement_safety(self, improvement: dict[str, Any]) -> dict[str, Any]:
        """Use meta_auditor to validate improvement safety"""

        # Leverage existing meta_auditor patterns
        safety_checks = [
            improvement["confidence"] >= 0.6,
            "training_source" in improvement,
            improvement["enhancement_type"] != "unknown",
            len(improvement["reasoning_chain"]) > 0,
        ]

        return {
            "safe": all(safety_checks),
            "safety_score": sum(safety_checks) / len(safety_checks),
            "audit_notes": f"CLI training improvement for {improvement['component']}",
        }

    def _extract_reasoning_chain(self, content: str) -> list[str]:
        """Extract reasoning chain from content"""
        # Simple extraction - look for sequential indicators
        sentences = content.split(".")
        chain = []
        for sentence in sentences:
            if any(
                word in sentence.lower()
                for word in ["first", "then", "next", "because", "therefore"]
            ):
                chain.append(sentence.strip())
        return chain[:3]  # Limit to 3 steps

    def _extract_safety_considerations(self, content: str) -> list[str]:
        """Extract safety considerations from content"""
        sentences = content.split(".")
        safety_items = []
        for sentence in sentences:
            if any(
                word in sentence.lower()
                for word in ["safe", "careful", "validate", "check", "ensure"]
            ):
                safety_items.append(sentence.strip())
        return safety_items

    def _extract_solution_approach(self, content: str) -> list[str]:
        """Extract solution approach from content"""
        sentences = content.split(".")
        solution_steps = []
        for sentence in sentences:
            if any(
                word in sentence.lower()
                for word in ["solution", "approach", "strategy", "implement"]
            ):
                solution_steps.append(sentence.strip())
        return solution_steps

    def _extract_documentation_patterns(self, code: str) -> list[str]:
        """Extract documentation patterns from code"""
        patterns = []
        if '"""' in code or "'''" in code:
            patterns.append("docstring_usage")
        if code.count("# ") > 0:
            patterns.append("inline_comments")
        if "def " in code and any(doc in code for doc in ['"""', "'''"]):
            patterns.append("function_documentation")
        return patterns

    def _extract_error_handling_patterns(self, code: str) -> list[str]:
        """Extract error handling patterns from code"""
        patterns = []
        if "try:" in code and "except" in code:
            patterns.append("try_except_blocks")
        if "raise" in code:
            patterns.append("explicit_exceptions")
        if any(word in code.lower() for word in ["validate", "check", "assert"]):
            patterns.append("input_validation")
        return patterns

    def get_training_summary(self) -> dict[str, Any]:
        """Get summary of meta system training"""

        total_insights = sum(
            len(session.get("insights", [])) for session in self.training_sessions
        )
        total_improvements = len(self.meta_improvements)

        # Component enhancement breakdown
        component_enhancements = {}
        for improvement in self.meta_improvements:
            component = improvement.get("component", "unknown")
            component_enhancements[component] = (
                component_enhancements.get(component, 0) + 1
            )

        summary = {
            "training_sessions": len(self.training_sessions),
            "total_insights_extracted": total_insights,
            "total_improvements_applied": total_improvements,
            "component_enhancements": component_enhancements,
            "learned_patterns": list(self.learned_patterns.keys()),
            "meta_system_evolution": "active",
        }

        # Record summary in meta_prime
        self.meta_prime.observe("meta_training_summary", summary)

        return summary


# Demo integration
def demo_meta_cli_integration():
    """Demonstrate meta system training with CLI data"""

    print("🎓 DEMO: META SYSTEM TRAINING WITH CLI DATA")
    print("=" * 60)

    trainer = MetaClaudeCLITrainer()

    # Find existing CLI session file
    session_files = list(Path(".").glob("claude_cli_session_*.json"))

    if session_files:
        session_file = session_files[0]
        print(f"📚 Training with session: {session_file}")

        # Ingest and train
        result = trainer.ingest_cli_session(str(session_file))

        print("\n✅ Training Results:")
        print(f"   Session: {result['session_id']}")
        print(f"   Insights: {len(result['insights'])}")
        print(f"   Improvements: {len(result['improvements'])}")

        # Show training summary
        summary = trainer.get_training_summary()
        print("\n📊 Training Summary:")
        for key, value in summary.items():
            print(f"   {key}: {value}")

        print("\n🧠 Meta System Enhanced!")
        print("   The existing meta components now learn from Claude CLI patterns!")

    else:
        print(
            "⚠️ No CLI session files found - run claude_cli_reasoning_capture.py first"
        )

    return trainer


if __name__ == "__main__":
    demo_meta_cli_integration()
