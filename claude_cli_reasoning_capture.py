#!/usr/bin/env python3
"""
Claude CLI Reasoning Capture System
Captures Claude's reasoning patterns from CLI interactions for meta model training
"""

import time
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from meta_prime import MetaPrime
from meta_coordinator import MetaCoordinator


@dataclass
class CLIReasoningEvent:
    """Reasoning event captured from CLI interaction"""
    timestamp: float
    event_type: str  # 'user_request', 'claude_analysis', 'code_generation', 'error_recovery'
    content: str
    context: Dict[str, Any]
    reasoning_patterns: List[str]
    confidence: float = 0.8
    session_id: str = ""


class ClaudeCLIReasoningCapture:
    """Captures Claude's reasoning patterns from CLI interactions"""
    
    def __init__(self):
        self.meta_prime = MetaPrime()
        self.meta_coordinator = MetaCoordinator()
        
        # Session tracking
        self.current_session = f"cli_session_{int(time.time())}"
        self.captured_events: List[CLIReasoningEvent] = []
        self.reasoning_patterns = {}
        
        # Analysis state
        self.conversation_context = []
        self.code_generation_patterns = []
        self.problem_solving_chains = []
        
        print("🧠 Claude CLI Reasoning Capture initialized")
        print(f"📝 Session ID: {self.current_session}")
    
    def capture_user_request(self, request: str, context: Dict[str, Any] = None):
        """Capture user request that triggers Claude reasoning"""
        
        event = CLIReasoningEvent(
            timestamp=time.time(),
            event_type="user_request",
            content=request,
            context=context or {},
            reasoning_patterns=self._detect_request_patterns(request),
            session_id=self.current_session
        )
        
        self.captured_events.append(event)
        self.conversation_context.append(("user", request))
        
        # Record in meta system
        self.meta_prime.observe("cli_user_request", {
            "session_id": self.current_session,
            "request_length": len(request),
            "patterns": event.reasoning_patterns,
            "context_keys": list(event.context.keys())
        })
        
        print(f"📥 User Request: {request[:50]}...")
        return event
    
    def capture_claude_analysis(self, analysis: str, context: Dict[str, Any] = None):
        """Capture Claude's analysis and reasoning process"""
        
        reasoning_patterns = self._extract_reasoning_patterns(analysis)
        
        event = CLIReasoningEvent(
            timestamp=time.time(),
            event_type="claude_analysis", 
            content=analysis,
            context=context or {},
            reasoning_patterns=reasoning_patterns,
            confidence=self._assess_reasoning_confidence(analysis),
            session_id=self.current_session
        )
        
        self.captured_events.append(event)
        self.conversation_context.append(("claude_analysis", analysis))
        
        # Record patterns for meta learning
        for pattern in reasoning_patterns:
            self.reasoning_patterns[pattern] = self.reasoning_patterns.get(pattern, 0) + 1
        
        self.meta_prime.observe("cli_claude_analysis", {
            "session_id": self.current_session,
            "analysis_length": len(analysis),
            "reasoning_patterns": reasoning_patterns,
            "confidence": event.confidence,
            "pattern_frequency": self.reasoning_patterns
        })
        
        print(f"🧠 Claude Analysis: {len(reasoning_patterns)} patterns detected")
        return event
    
    def capture_code_generation(self, code: str, language: str, context: Dict[str, Any] = None):
        """Capture Claude's code generation with reasoning context"""
        
        code_patterns = self._analyze_code_patterns(code, language)
        
        event = CLIReasoningEvent(
            timestamp=time.time(),
            event_type="code_generation",
            content=code,
            context={
                "language": language,
                "code_length": len(code),
                "lines": code.count('\n') + 1,
                **(context or {})
            },
            reasoning_patterns=code_patterns,
            confidence=self._assess_code_quality(code),
            session_id=self.current_session
        )
        
        self.captured_events.append(event)
        self.code_generation_patterns.append({
            "patterns": code_patterns,
            "language": language,
            "context": event.context
        })
        
        self.meta_prime.observe("cli_code_generation", {
            "session_id": self.current_session,
            "language": language,
            "code_patterns": code_patterns,
            "quality_score": event.confidence,
            "generation_context": event.context
        })
        
        print(f"💻 Code Generated: {language}, {len(code_patterns)} patterns")
        return event
    
    def capture_error_recovery(self, error: str, solution: str, context: Dict[str, Any] = None):
        """Capture Claude's error analysis and recovery strategy"""
        
        recovery_patterns = self._analyze_error_recovery(error, solution)
        
        event = CLIReasoningEvent(
            timestamp=time.time(),
            event_type="error_recovery",
            content=f"ERROR: {error}\nSOLUTION: {solution}",
            context={
                "error_type": self._classify_error(error),
                "solution_approach": self._classify_solution(solution),
                **(context or {})
            },
            reasoning_patterns=recovery_patterns,
            confidence=self._assess_solution_confidence(solution),
            session_id=self.current_session
        )
        
        self.captured_events.append(event)
        self.problem_solving_chains.append({
            "error": error,
            "solution": solution,
            "patterns": recovery_patterns,
            "context": event.context
        })
        
        self.meta_prime.observe("cli_error_recovery", {
            "session_id": self.current_session,
            "error_type": event.context["error_type"],
            "solution_approach": event.context["solution_approach"],
            "recovery_patterns": recovery_patterns,
            "confidence": event.confidence
        })
        
        print(f"🔧 Error Recovery: {event.context['error_type']} → {event.context['solution_approach']}")
        return event
    
    def _detect_request_patterns(self, request: str) -> List[str]:
        """Detect patterns in user requests"""
        patterns = []
        request_lower = request.lower()
        
        # Request type patterns
        if any(word in request_lower for word in ["implement", "create", "build", "add"]):
            patterns.append("creation_request")
        if any(word in request_lower for word in ["fix", "debug", "error", "problem"]):
            patterns.append("debugging_request")
        if any(word in request_lower for word in ["test", "validate", "verify", "check"]):
            patterns.append("validation_request")
        if any(word in request_lower for word in ["optimize", "improve", "enhance", "better"]):
            patterns.append("optimization_request")
        if any(word in request_lower for word in ["explain", "how", "why", "what"]):
            patterns.append("explanation_request")
            
        # Complexity indicators
        if len(request.split()) > 20:
            patterns.append("complex_request")
        if any(word in request_lower for word in ["system", "architecture", "integration"]):
            patterns.append("system_level_request")
            
        return patterns
    
    def _extract_reasoning_patterns(self, analysis: str) -> List[str]:
        """Extract reasoning patterns from Claude's analysis"""
        patterns = []
        analysis_lower = analysis.lower()
        
        # Reasoning approach patterns
        if any(phrase in analysis_lower for phrase in ["step by step", "systematically", "methodically"]):
            patterns.append("systematic_reasoning")
        if any(phrase in analysis_lower for phrase in ["first", "then", "next", "finally"]):
            patterns.append("sequential_reasoning")
        if any(phrase in analysis_lower for phrase in ["because", "since", "due to", "therefore"]):
            patterns.append("causal_reasoning")
        if any(phrase in analysis_lower for phrase in ["however", "but", "although", "despite"]):
            patterns.append("contrastive_reasoning")
            
        # Problem-solving patterns
        if any(word in analysis_lower for word in ["analyze", "examine", "investigate"]):
            patterns.append("analytical_approach")
        if any(word in analysis_lower for word in ["solution", "approach", "strategy", "method"]):
            patterns.append("solution_oriented")
        if any(word in analysis_lower for word in ["consider", "option", "alternative"]):
            patterns.append("option_evaluation")
            
        # Safety and validation patterns
        if any(word in analysis_lower for word in ["safe", "careful", "validate", "verify"]):
            patterns.append("safety_conscious")
        if any(word in analysis_lower for word in ["test", "check", "ensure", "confirm"]):
            patterns.append("validation_focused")
            
        return patterns
    
    def _analyze_code_patterns(self, code: str, language: str) -> List[str]:
        """Analyze patterns in generated code"""
        patterns = []
        
        # Language-specific patterns
        if language.lower() == "python":
            if "async def" in code:
                patterns.append("async_programming")
            if "try:" in code and "except" in code:
                patterns.append("error_handling")
            if "class " in code:
                patterns.append("object_oriented")
            if "import " in code:
                patterns.append("modular_design")
        
        # General coding patterns
        if any(word in code.lower() for word in ["print(", "log", "debug"]):
            patterns.append("debugging_oriented")
        if code.count('\n') > 20:
            patterns.append("comprehensive_implementation")
        if any(word in code for word in ["# ", "\"\"\"", "'''"]):
            patterns.append("well_documented")
        if any(word in code.lower() for word in ["validate", "check", "assert"]):
            patterns.append("validation_included")
            
        return patterns
    
    def _analyze_error_recovery(self, error: str, solution: str) -> List[str]:
        """Analyze patterns in error recovery approaches"""
        patterns = []
        
        # Error analysis patterns
        if "import" in error.lower():
            patterns.append("import_error_recovery")
        if "syntax" in error.lower():
            patterns.append("syntax_error_recovery")
        if "attribute" in error.lower():
            patterns.append("attribute_error_recovery")
        if "type" in error.lower():
            patterns.append("type_error_recovery")
            
        # Solution approach patterns
        if "install" in solution.lower():
            patterns.append("dependency_solution")
        if "fix" in solution.lower():
            patterns.append("direct_fix_approach")
        if "alternative" in solution.lower():
            patterns.append("alternative_approach")
        if "check" in solution.lower():
            patterns.append("validation_solution")
            
        return patterns
    
    def _assess_reasoning_confidence(self, analysis: str) -> float:
        """Assess confidence in reasoning quality"""
        confidence = 0.5  # Base confidence
        
        # Boost for structured reasoning
        if any(word in analysis.lower() for word in ["first", "second", "then", "finally"]):
            confidence += 0.2
        if any(word in analysis.lower() for word in ["because", "therefore", "since"]):
            confidence += 0.1
        if len(analysis.split()) > 50:  # Detailed analysis
            confidence += 0.1
        if any(word in analysis.lower() for word in ["test", "verify", "validate"]):
            confidence += 0.1
            
        return min(confidence, 1.0)
    
    def _assess_code_quality(self, code: str) -> float:
        """Assess quality of generated code"""
        quality = 0.5  # Base quality
        
        # Quality indicators
        if "try:" in code and "except" in code:
            quality += 0.1  # Error handling
        if any(word in code for word in ["# ", "\"\"\"", "'''"]):
            quality += 0.1  # Documentation
        if code.count('\n') > 10:
            quality += 0.1  # Comprehensive
        if any(word in code.lower() for word in ["validate", "check", "assert"]):
            quality += 0.1  # Validation
        if "import " in code:
            quality += 0.1  # Proper imports
            
        return min(quality, 1.0)
    
    def _assess_solution_confidence(self, solution: str) -> float:
        """Assess confidence in solution effectiveness"""
        confidence = 0.5
        
        if len(solution.split()) > 20:  # Detailed solution
            confidence += 0.2
        if any(word in solution.lower() for word in ["test", "verify", "check"]):
            confidence += 0.1
        if "example" in solution.lower():
            confidence += 0.1
        if any(word in solution.lower() for word in ["should", "will", "ensure"]):
            confidence += 0.1
            
        return min(confidence, 1.0)
    
    def _classify_error(self, error: str) -> str:
        """Classify type of error"""
        error_lower = error.lower()
        
        if "import" in error_lower:
            return "import_error"
        elif "syntax" in error_lower:
            return "syntax_error"
        elif "attribute" in error_lower:
            return "attribute_error"
        elif "type" in error_lower:
            return "type_error"
        elif "file" in error_lower:
            return "file_error"
        else:
            return "other_error"
    
    def _classify_solution(self, solution: str) -> str:
        """Classify type of solution approach"""
        solution_lower = solution.lower()
        
        if "install" in solution_lower:
            return "dependency_fix"
        elif "change" in solution_lower or "modify" in solution_lower:
            return "code_modification"
        elif "add" in solution_lower:
            return "code_addition"
        elif "remove" in solution_lower:
            return "code_removal"
        elif "check" in solution_lower:
            return "validation_approach"
        else:
            return "other_solution"
    
    def generate_training_insights(self) -> Dict[str, Any]:
        """Generate insights for meta model training"""
        
        total_events = len(self.captured_events)
        if total_events == 0:
            return {}
        
        # Pattern frequency analysis
        all_patterns = []
        for event in self.captured_events:
            all_patterns.extend(event.reasoning_patterns)
        
        pattern_frequency = {}
        for pattern in all_patterns:
            pattern_frequency[pattern] = all_patterns.count(pattern)
        
        # Event type distribution
        event_types = [event.event_type for event in self.captured_events]
        event_distribution = {}
        for event_type in set(event_types):
            event_distribution[event_type] = event_types.count(event_type)
        
        # Quality metrics
        avg_confidence = sum(event.confidence for event in self.captured_events) / total_events
        
        insights = {
            "session_id": self.current_session,
            "total_events": total_events,
            "pattern_frequency": pattern_frequency,
            "event_distribution": event_distribution,
            "avg_confidence": avg_confidence,
            "conversation_length": len(self.conversation_context),
            "code_generations": len(self.code_generation_patterns),
            "problem_solving_chains": len(self.problem_solving_chains),
            "top_patterns": sorted(pattern_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        }
        
        # Record insights in meta system
        self.meta_prime.observe("cli_training_insights", insights)
        
        return insights
    
    def save_session_data(self, filepath: Optional[str] = None):
        """Save captured session data for training"""
        
        if not filepath:
            filepath = f"claude_cli_session_{self.current_session}.json"
        
        session_data = {
            "session_id": self.current_session,
            "timestamp": time.time(),
            "events": [asdict(event) for event in self.captured_events],
            "conversation_context": self.conversation_context,
            "code_generation_patterns": self.code_generation_patterns,
            "problem_solving_chains": self.problem_solving_chains,
            "reasoning_patterns": self.reasoning_patterns,
            "insights": self.generate_training_insights()
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"💾 Session data saved: {filepath}")
        return filepath


# Demo: Capture current conversation
def demo_current_conversation():
    """Demonstrate capturing this current conversation"""
    
    print("🎯 DEMO: CAPTURING CURRENT CLAUDE CLI CONVERSATION")
    print("=" * 60)
    
    capture = ClaudeCLIReasoningCapture()
    
    # Simulate capturing this conversation
    capture.capture_user_request(
        "since we can't capture coming in, are we still capturing sort of what we see at the CLI level as a precursor to the coding to give it feedback or at least to train our /meta model",
        {"domain": "meta_learning", "complexity": "high"}
    )
    
    capture.capture_claude_analysis(
        "Absolutely! You've identified the perfect fallback strategy. We can capture Claude's reasoning at the CLI level - which is exactly what's happening in this conversation right now. This gives us valuable training data for the meta model including problem-solving approaches, decision-making patterns, and code generation logic.",
        {"reasoning_depth": "high", "solution_oriented": True}
    )
    
    capture.capture_code_generation(
        """#!/usr/bin/env python3
def demo_current_conversation():
    capture = ClaudeCLIReasoningCapture()
    # Capture reasoning patterns from CLI interaction
    return capture""",
        "python",
        {"purpose": "demonstration", "complexity": "medium"}
    )
    
    # Generate insights
    insights = capture.generate_training_insights()
    
    print(f"\n📊 CONVERSATION INSIGHTS:")
    for key, value in insights.items():
        if isinstance(value, (int, float, str)):
            print(f"   {key}: {value}")
    
    print(f"\n🧠 TOP REASONING PATTERNS:")
    for pattern, count in insights["top_patterns"]:
        print(f"   • {pattern}: {count} occurrences")
    
    # Save session
    filepath = capture.save_session_data()
    
    print(f"\n✅ Demo complete - this conversation captured as training data!")
    return capture


if __name__ == "__main__":
    demo_current_conversation()