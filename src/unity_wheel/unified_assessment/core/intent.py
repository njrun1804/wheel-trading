"""
Intent Analysis Layer - Multi-model intent classification with context awareness.

This module analyzes user intent by combining:
- Pattern-based classification
- Context-aware disambiguation
- Confidence scoring
- Parameter extraction
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..schemas.command import IntentCategory, Target, Constraint
from .context import GatheredContext

logger = logging.getLogger(__name__)


@dataclass
class Intent:
    """Parsed user intent with confidence scoring."""
    
    category: IntentCategory
    action: str  # Specific action verb (fix, create, optimize, analyze, etc.)
    targets: List[Target]  # What to act upon
    constraints: List[Constraint]  # Limitations or requirements
    confidence: float  # 0.0 to 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""  # Why this intent was selected


@dataclass
class IntentAnalysis:
    """Complete intent analysis result."""
    
    primary_intent: Intent
    alternative_intents: List[Intent] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires_clarification: bool = False
    clarification_questions: List[str] = field(default_factory=list)
    analysis_confidence: float = 0.0
    processing_time_ms: float = 0.0


class IntentPatternMatcher:
    """Pattern-based intent classification using rules and heuristics."""
    
    def __init__(self):
        self.intent_patterns = {
            IntentCategory.FIX: [
                (r'\b(fix|repair|resolve|debug|correct|solve)\b', 0.9),
                (r'\b(error|bug|issue|problem|broken|failing)\b', 0.8),
                (r'\b(not working|doesn\'t work|broken)\b', 0.8),
                (r'\b(exception|crash|fail)\b', 0.7),
            ],
            IntentCategory.CREATE: [
                (r'\b(create|add|build|make|generate|implement)\b', 0.9),
                (r'\b(new|fresh|blank)\b', 0.7),
                (r'\b(setup|initialize|scaffold)\b', 0.8),
            ],
            IntentCategory.OPTIMIZE: [
                (r'\b(optimize|improve|enhance|speed up|faster)\b', 0.9),
                (r'\b(performance|efficiency|slow|bottleneck)\b', 0.8),
                (r'\b(refactor|streamline|cleanup)\b', 0.7),
            ],
            IntentCategory.ANALYZE: [
                (r'\b(analyze|examine|review|inspect|check)\b', 0.9),
                (r'\b(what|how|why|explain|show me)\b', 0.6),
                (r'\b(understand|explore|investigate)\b', 0.8),
            ],
            IntentCategory.REFACTOR: [
                (r'\b(refactor|restructure|reorganize|clean up)\b', 0.9),
                (r'\b(improve code|better structure)\b', 0.8),
                (r'\b(duplicate|repetitive|messy)\b', 0.7),
            ],
            IntentCategory.TEST: [
                (r'\b(test|testing|unit test|integration test)\b', 0.9),
                (r'\b(coverage|spec|validation)\b', 0.7),
                (r'\b(mock|stub|fixture)\b', 0.8),
            ],
            IntentCategory.DEPLOY: [
                (r'\b(deploy|deployment|release|publish)\b', 0.9),
                (r'\b(production|staging|environment)\b', 0.7),
                (r'\b(ci/cd|pipeline|build)\b', 0.8),
            ],
            IntentCategory.MONITOR: [
                (r'\b(monitor|monitoring|observe|track)\b', 0.9),
                (r'\b(metrics|logs|alerts|dashboard)\b', 0.8),
                (r'\b(health|status|uptime)\b', 0.7),
            ],
            IntentCategory.QUERY: [
                (r'\b(show|list|find|search|what is|where is)\b', 0.8),
                (r'\b(get|fetch|retrieve|display)\b', 0.7),
                (r'\b(status|info|details)\b', 0.6),
            ]
        }
        
        self.action_patterns = {
            'fix': [r'\b(fix|repair|resolve|debug|correct)\b'],
            'create': [r'\b(create|add|build|make|generate)\b'],
            'optimize': [r'\b(optimize|improve|enhance|speed up)\b'],
            'analyze': [r'\b(analyze|examine|review|inspect)\b'],
            'refactor': [r'\b(refactor|restructure|reorganize)\b'],
            'test': [r'\b(test|testing|validate)\b'],
            'deploy': [r'\b(deploy|release|publish)\b'],
            'monitor': [r'\b(monitor|track|observe)\b'],
            'query': [r'\b(show|list|find|get|what|where)\b'],
        }
        
        self.target_patterns = {
            'file': [r'\.py\b', r'\.js\b', r'\.ts\b', r'\.json\b', r'\.yaml\b', r'\.md\b'],
            'function': [r'\bfunction\b', r'\bdef\b', r'\bmethod\b'],
            'class': [r'\bclass\b', r'\bcomponent\b'],
            'module': [r'\bmodule\b', r'\bpackage\b', r'\blibrary\b'],
            'system': [r'\bsystem\b', r'\bservice\b', r'\bapplication\b', r'\bapi\b'],
            'database': [r'\bdatabase\b', r'\bdb\b', r'\bsql\b', r'\btable\b'],
            'authentication': [r'\bauth\b', r'\blogin\b', r'\btoken\b', r'\bsecurity\b'],
            'performance': [r'\bperformance\b', r'\bspeed\b', r'\bmemory\b', r'\bcpu\b'],
            'error': [r'\berror\b', r'\bexception\b', r'\bbug\b', r'\bissue\b'],
        }
    
    def match_intent(self, command: str) -> List[tuple[IntentCategory, float]]:
        """Match command against intent patterns."""
        command_lower = command.lower()
        matches = []
        
        for category, patterns in self.intent_patterns.items():
            max_score = 0.0
            for pattern, base_score in patterns:
                if re.search(pattern, command_lower):
                    max_score = max(max_score, base_score)
            
            if max_score > 0:
                matches.append((category, max_score))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)
    
    def extract_action(self, command: str) -> str:
        """Extract the primary action from the command."""
        command_lower = command.lower()
        
        for action, patterns in self.action_patterns.items():
            for pattern in patterns:
                if re.search(pattern, command_lower):
                    return action
        
        return "unknown"
    
    def extract_targets(self, command: str, context: GatheredContext) -> List[Target]:
        """Extract targets from command using context awareness."""
        targets = []
        command_lower = command.lower()
        
        # Extract file targets from command
        for file_context in context.relevant_files[:5]:  # Top 5 most relevant
            file_name = file_context.file_path.split('/')[-1]
            if file_name.lower() in command_lower:
                targets.append(Target(
                    target_type="file",
                    identifier=file_context.file_path,
                    location=file_context.file_path,
                    confidence=file_context.relevance_score
                ))
        
        # Extract system/component targets
        for target_type, patterns in self.target_patterns.items():
            for pattern in patterns:
                if re.search(pattern, command_lower):
                    targets.append(Target(
                        target_type=target_type,
                        identifier=target_type,
                        confidence=0.7
                    ))
                    break
        
        return targets
    
    def extract_constraints(self, command: str) -> List[Constraint]:
        """Extract constraints from command."""
        constraints = []
        command_lower = command.lower()
        
        constraint_patterns = {
            'compatibility': [r'\bcompatible\b', r'\bbackward\b', r'\bbreaking\b'],
            'performance': [r'\bfast\b', r'\bquick\b', r'\befficient\b', r'\bslow\b'],
            'safety': [r'\bsafe\b', r'\bsecure\b', r'\bvalidate\b', r'\btest\b'],
            'style': [r'\bstyle\b', r'\bformat\b', r'\bclean\b', r'\breadable\b'],
        }
        
        for constraint_type, patterns in constraint_patterns.items():
            for pattern in patterns:
                if re.search(pattern, command_lower):
                    constraints.append(Constraint(
                        constraint_type=constraint_type,
                        description=f"Maintain {constraint_type} requirements",
                        severity="medium"
                    ))
                    break
        
        return constraints


class ContextualIntentRefiner:
    """Refine intent analysis using context information."""
    
    def refine_intent(self, intent: Intent, context: GatheredContext) -> Intent:
        """Refine intent using gathered context."""
        
        # Boost confidence if we have relevant files
        if context.relevant_files and intent.category in [
            IntentCategory.FIX, IntentCategory.OPTIMIZE, IntentCategory.REFACTOR
        ]:
            intent.confidence = min(intent.confidence + 0.1, 1.0)
            intent.reasoning += f" (boosted by {len(context.relevant_files)} relevant files)"
        
        # Adjust targets based on context
        if not intent.targets and context.relevant_files:
            # Add top relevant files as targets
            for file_context in context.relevant_files[:3]:
                intent.targets.append(Target(
                    target_type="file",
                    identifier=file_context.file_path,
                    location=file_context.file_path,
                    confidence=file_context.relevance_score
                ))
        
        # Add constraints based on system state
        if context.system_state.uncommitted_changes:
            intent.constraints.append(Constraint(
                constraint_type="safety",
                description="Handle uncommitted changes carefully",
                severity="medium"
            ))
        
        # Add performance constraints for optimization intents
        if intent.category == IntentCategory.OPTIMIZE:
            intent.constraints.append(Constraint(
                constraint_type="performance",
                description="Measure performance impact",
                severity="high"
            ))
        
        return intent


class IntentAnalyzer:
    """
    Intent analyzer that combines pattern matching with context awareness.
    
    Analyzes user commands to understand what they want to achieve,
    using both linguistic patterns and codebase context.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pattern_matcher = IntentPatternMatcher()
        self.contextual_refiner = ContextualIntentRefiner()
        
        # Performance tracking
        self.total_analyses = 0
        self.average_analysis_time_ms = 0.0
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize intent analyzer."""
        if self.initialized:
            return
        
        logger.info("🧠 Initializing Intent Analyzer...")
        
        # Intent analysis is mostly rule-based, so initialization is quick
        self.initialized = True
        
        logger.info("✅ Intent Analyzer initialized")
    
    async def analyze_intent(
        self,
        command: str,
        context: GatheredContext
    ) -> IntentAnalysis:
        """
        Analyze user intent with full context awareness.
        
        Args:
            command: Natural language command
            context: Gathered context from context layer
            
        Returns:
            IntentAnalysis with primary intent and alternatives
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        try:
            # Stage 1: Pattern-based classification
            intent_matches = self.pattern_matcher.match_intent(command)
            
            if not intent_matches:
                # Fallback to unknown intent
                intent_matches = [(IntentCategory.UNKNOWN, 0.5)]
            
            # Stage 2: Create primary intent
            primary_category, primary_confidence = intent_matches[0]
            
            primary_intent = Intent(
                category=primary_category,
                action=self.pattern_matcher.extract_action(command),
                targets=self.pattern_matcher.extract_targets(command, context),
                constraints=self.pattern_matcher.extract_constraints(command),
                confidence=primary_confidence,
                reasoning=f"Matched pattern for {primary_category.value}"
            )
            
            # Stage 3: Context-aware refinement
            primary_intent = self.contextual_refiner.refine_intent(primary_intent, context)
            
            # Stage 4: Generate alternatives
            alternative_intents = []
            for category, confidence in intent_matches[1:3]:  # Top 2 alternatives
                alt_intent = Intent(
                    category=category,
                    action=self.pattern_matcher.extract_action(command),
                    targets=primary_intent.targets,  # Reuse targets
                    constraints=primary_intent.constraints,  # Reuse constraints
                    confidence=confidence * 0.8,  # Reduce confidence for alternatives
                    reasoning=f"Alternative interpretation as {category.value}"
                )
                alternative_intents.append(alt_intent)
            
            # Stage 5: Parameter extraction
            parameters = self._extract_parameters(command, context, primary_intent)
            
            # Stage 6: Clarification analysis
            requires_clarification, clarification_questions = self._analyze_clarification_needs(
                primary_intent, alternative_intents, context
            )
            
            # Calculate overall analysis confidence
            analysis_confidence = self._calculate_analysis_confidence(
                primary_intent, alternative_intents, context
            )
            
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Create analysis result
            analysis = IntentAnalysis(
                primary_intent=primary_intent,
                alternative_intents=alternative_intents,
                parameters=parameters,
                requires_clarification=requires_clarification,
                clarification_questions=clarification_questions,
                analysis_confidence=analysis_confidence,
                processing_time_ms=processing_time_ms
            )
            
            # Update statistics
            self.total_analyses += 1
            self.average_analysis_time_ms = (
                (self.average_analysis_time_ms * (self.total_analyses - 1) + processing_time_ms)
                / self.total_analyses
            )
            
            logger.debug(
                f"🧠 Intent analyzed in {processing_time_ms:.1f}ms "
                f"(category: {primary_intent.category.value}, "
                f"confidence: {primary_intent.confidence:.2f})"
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Intent analysis failed: {e}")
            
            # Return fallback analysis
            fallback_intent = Intent(
                category=IntentCategory.UNKNOWN,
                action="unknown",
                targets=[],
                constraints=[],
                confidence=0.0,
                reasoning="Analysis failed, using fallback"
            )
            
            return IntentAnalysis(
                primary_intent=fallback_intent,
                alternative_intents=[],
                parameters={},
                requires_clarification=True,
                clarification_questions=[
                    "Could you please rephrase your command?",
                    "What specific action would you like to take?"
                ],
                analysis_confidence=0.0,
                processing_time_ms=(time.perf_counter() - start_time) * 1000
            )
    
    def _extract_parameters(
        self,
        command: str,
        context: GatheredContext,
        intent: Intent
    ) -> Dict[str, Any]:
        """Extract parameters from command for the given intent."""
        parameters = {}
        
        # Extract common parameters
        parameters['command_length'] = len(command)
        parameters['has_file_references'] = len(intent.targets) > 0
        parameters['context_confidence'] = context.confidence_score
        parameters['relevant_files_count'] = len(context.relevant_files)
        
        # Category-specific parameters
        if intent.category == IntentCategory.FIX:
            parameters['error_keywords'] = [
                word for word in ['error', 'bug', 'issue', 'problem', 'exception']
                if word in command.lower()
            ]
        
        elif intent.category == IntentCategory.CREATE:
            parameters['creation_type'] = 'file' if any(
                ext in command for ext in ['.py', '.js', '.ts', '.json']
            ) else 'component'
        
        elif intent.category == IntentCategory.OPTIMIZE:
            parameters['optimization_target'] = 'performance' if any(
                word in command.lower() for word in ['fast', 'slow', 'speed', 'performance']
            ) else 'code'
        
        return parameters
    
    def _analyze_clarification_needs(
        self,
        primary_intent: Intent,
        alternatives: List[Intent],
        context: GatheredContext
    ) -> tuple[bool, List[str]]:
        """Analyze if clarification is needed."""
        
        questions = []
        
        # Low confidence requires clarification
        if primary_intent.confidence < 0.6:
            questions.append("I'm not completely sure what you want to do. Could you be more specific?")
        
        # Multiple high-confidence alternatives
        if len(alternatives) > 0 and alternatives[0].confidence > 0.7:
            questions.append(
                f"Do you want to {primary_intent.action} or {alternatives[0].action}?"
            )
        
        # No targets identified
        if not primary_intent.targets and primary_intent.category not in [
            IntentCategory.QUERY, IntentCategory.ANALYZE
        ]:
            questions.append("What specific files or components should I work with?")
        
        # Ambiguous constraints
        if len(primary_intent.constraints) == 0 and primary_intent.category in [
            IntentCategory.REFACTOR, IntentCategory.OPTIMIZE
        ]:
            questions.append("Are there any specific requirements or constraints I should consider?")
        
        return len(questions) > 0, questions
    
    def _calculate_analysis_confidence(
        self,
        primary_intent: Intent,
        alternatives: List[Intent],
        context: GatheredContext
    ) -> float:
        """Calculate overall confidence in the intent analysis."""
        
        # Base confidence from primary intent
        confidence = primary_intent.confidence * 0.6
        
        # Boost if we have good context
        if context.confidence_score > 0.7:
            confidence += 0.2
        
        # Boost if targets are clearly identified
        if primary_intent.targets:
            confidence += 0.1
        
        # Reduce if there are strong alternatives
        if alternatives and alternatives[0].confidence > 0.7:
            confidence -= 0.1
        
        return min(max(confidence, 0.0), 1.0)
    
    async def shutdown(self):
        """Shutdown intent analyzer."""
        logger.debug("🔄 Shutting down Intent Analyzer")
        self.initialized = False
        logger.debug("✅ Intent Analyzer shutdown complete")