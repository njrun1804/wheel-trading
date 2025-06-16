"""
Main Unified Assessment Engine - Single entry point for all natural language commands.

The UAE processes commands through a four-stage pipeline:
1. Context Gathering - Einstein semantic search and code analysis
2. Intent Analysis - Multi-model intent classification with context awareness  
3. Action Planning - Task decomposition and resource optimization
4. Execution Routing - Bolt multi-agent and direct tool coordination
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from ..schemas.command import CommandResult, CommandStatus, CommandMetrics
from .context import ContextGatherer, GatheredContext
from .intent import IntentAnalyzer, IntentAnalysis
from .planning import ActionPlanner, ExecutionPlan
from .routing import ExecutionRouter

logger = logging.getLogger(__name__)


class UnifiedAssessmentEngine:
    """
    Main engine that coordinates all components of the unified assessment system.
    
    Provides a single entry point for processing natural language commands
    with comprehensive context awareness and optimized execution.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the unified assessment engine."""
        self.config = config or {}
        
        # Core components (initialized lazily)
        self._context_gatherer: Optional[ContextGatherer] = None
        self._intent_analyzer: Optional[IntentAnalyzer] = None
        self._action_planner: Optional[ActionPlanner] = None
        self._execution_router: Optional[ExecutionRouter] = None
        
        # Performance tracking
        self.total_commands_processed = 0
        self.successful_commands = 0
        self.average_processing_time_ms = 0.0
        
        # Engine state
        self.initialized = False
        self.engine_id = f"uae_{int(time.time())}"
        
        logger.info(f"🚀 Unified Assessment Engine created: {self.engine_id}")
    
    async def initialize(self):
        """Initialize all engine components."""
        if self.initialized:
            return
        
        start_time = time.perf_counter()
        logger.info("🔧 Initializing Unified Assessment Engine components...")
        
        try:
            # Initialize components in parallel for faster startup
            init_tasks = [
                self._init_context_gatherer(),
                self._init_intent_analyzer(),
                self._init_action_planner(),
                self._init_execution_router()
            ]
            
            await asyncio.gather(*init_tasks)
            
            self.initialized = True
            init_time_ms = (time.perf_counter() - start_time) * 1000
            
            logger.info(f"✅ UAE initialized successfully in {init_time_ms:.1f}ms")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize UAE: {e}")
            raise
    
    async def process_command(
        self,
        command: str,
        context_hints: Optional[Dict[str, Any]] = None,
        optimization_target: str = "balanced"
    ) -> CommandResult:
        """
        Process a natural language command through the unified pipeline.
        
        Args:
            command: Natural language command to process
            context_hints: Optional hints to guide context gathering
            optimization_target: "speed", "accuracy", or "balanced"
            
        Returns:
            CommandResult with processing results and metrics
        """
        if not self.initialized:
            await self.initialize()
        
        # Create result container
        result = CommandResult(original_command=command)
        result.status = CommandStatus.PENDING
        result.log_execution_step("command_received", {"command": command})
        
        start_time = time.perf_counter()
        
        try:
            # Stage 1: Gather Context
            result.status = CommandStatus.GATHERING_CONTEXT
            result.log_execution_step("context_gathering_started")
            
            context_start = time.perf_counter()
            context = await self._gather_context(command, context_hints)
            context_time_ms = (time.perf_counter() - context_start) * 1000
            result.metrics.context_gathering_ms = context_time_ms
            result.metrics.context_confidence = context.confidence_score
            
            result.log_execution_step("context_gathering_completed", {
                "duration_ms": context_time_ms,
                "confidence": context.confidence_score,
                "files_found": len(context.relevant_files)
            })
            
            # Stage 2: Analyze Intent
            result.status = CommandStatus.ANALYZING_INTENT
            result.log_execution_step("intent_analysis_started")
            
            intent_start = time.perf_counter()
            intent_analysis = await self._analyze_intent(command, context)
            intent_time_ms = (time.perf_counter() - intent_start) * 1000
            result.metrics.intent_analysis_ms = intent_time_ms
            result.metrics.intent_confidence = intent_analysis.primary_intent.confidence
            
            result.log_execution_step("intent_analysis_completed", {
                "duration_ms": intent_time_ms,
                "confidence": intent_analysis.primary_intent.confidence,
                "category": intent_analysis.primary_intent.category.value,
                "requires_clarification": intent_analysis.requires_clarification
            })
            
            # Handle clarification if needed
            if intent_analysis.requires_clarification:
                result.warnings.extend(intent_analysis.clarification_questions)
                result.add_recommendation("Please clarify your command for better results")
            
            # Stage 3: Plan Actions
            result.status = CommandStatus.PLANNING_ACTIONS
            result.log_execution_step("action_planning_started")
            
            planning_start = time.perf_counter()
            execution_plan = await self._plan_actions(intent_analysis, context)
            planning_time_ms = (time.perf_counter() - planning_start) * 1000
            result.metrics.action_planning_ms = planning_time_ms
            
            result.log_execution_step("action_planning_completed", {
                "duration_ms": planning_time_ms,
                "tasks_planned": len(execution_plan.tasks),
                "estimated_duration_ms": execution_plan.estimated_total_duration_ms
            })
            
            # Stage 4: Execute Plan
            result.status = CommandStatus.EXECUTING
            result.log_execution_step("execution_started")
            
            execution_start = time.perf_counter()
            execution_results = await self._execute_plan(execution_plan, context)
            execution_time_ms = (time.perf_counter() - execution_start) * 1000
            result.metrics.execution_ms = execution_time_ms
            
            result.log_execution_step("execution_completed", {
                "duration_ms": execution_time_ms,
                "success": execution_results.success
            })
            
            # Stage 5: Synthesize Results
            await self._synthesize_results(result, execution_results, context, intent_analysis)
            
            # Mark as completed
            result.mark_completed(success=execution_results.success)
            
            # Update engine statistics
            self.total_commands_processed += 1
            if result.success:
                self.successful_commands += 1
            
            total_time_ms = (time.perf_counter() - start_time) * 1000
            self.average_processing_time_ms = (
                (self.average_processing_time_ms * (self.total_commands_processed - 1) + total_time_ms)
                / self.total_commands_processed
            )
            
            logger.info(
                f"✅ Command processed successfully in {total_time_ms:.1f}ms "
                f"(Success rate: {self.successful_commands/self.total_commands_processed:.1%})"
            )
            
        except Exception as e:
            logger.error(f"❌ Command processing failed: {e}")
            result.add_error("processing_error", str(e))
            result.mark_completed(success=False)
        
        return result
    
    async def _init_context_gatherer(self):
        """Initialize the context gatherer component."""
        self._context_gatherer = ContextGatherer(self.config.get("context", {}))
        await self._context_gatherer.initialize()
    
    async def _init_intent_analyzer(self):
        """Initialize the intent analyzer component."""
        self._intent_analyzer = IntentAnalyzer(self.config.get("intent", {}))
        await self._intent_analyzer.initialize()
    
    async def _init_action_planner(self):
        """Initialize the action planner component."""
        self._action_planner = ActionPlanner(self.config.get("planning", {}))
        await self._action_planner.initialize()
    
    async def _init_execution_router(self):
        """Initialize the execution router component."""
        self._execution_router = ExecutionRouter(self.config.get("routing", {}))
        await self._execution_router.initialize()
    
    async def _gather_context(
        self, 
        command: str, 
        hints: Optional[Dict[str, Any]]
    ) -> GatheredContext:
        """Gather comprehensive context for the command."""
        return await self._context_gatherer.gather_context(command, hints)
    
    async def _analyze_intent(
        self, 
        command: str, 
        context: GatheredContext
    ) -> IntentAnalysis:
        """Analyze user intent with context awareness."""
        return await self._intent_analyzer.analyze_intent(command, context)
    
    async def _plan_actions(
        self, 
        intent_analysis: IntentAnalysis, 
        context: GatheredContext
    ) -> ExecutionPlan:
        """Create an optimized execution plan."""
        return await self._action_planner.plan_actions(intent_analysis, context)
    
    async def _execute_plan(
        self, 
        plan: ExecutionPlan, 
        context: GatheredContext
    ):
        """Execute the action plan using appropriate routing."""
        return await self._execution_router.execute_plan(plan, context)
    
    async def _synthesize_results(
        self, 
        result: CommandResult,
        execution_results,
        context: GatheredContext,
        intent_analysis: IntentAnalysis
    ):
        """Synthesize final results from all processing stages."""
        
        # Generate summary
        if execution_results.success:
            result.summary = f"Successfully {intent_analysis.primary_intent.action}"
            if execution_results.changes_made:
                result.summary += f" ({len(execution_results.changes_made)} changes made)"
        else:
            result.summary = f"Failed to {intent_analysis.primary_intent.action}"
            if execution_results.errors:
                result.summary += f" ({len(execution_results.errors)} errors encountered)"
        
        # Copy findings and recommendations
        result.findings.extend(execution_results.findings or [])
        result.recommendations.extend(execution_results.recommendations or [])
        result.actions_taken.extend(execution_results.actions_taken or [])
        
        # Copy file information
        result.files_affected.extend(execution_results.files_affected or [])
        result.changes_made.extend(execution_results.changes_made or [])
        
        # Copy errors and metrics
        if hasattr(execution_results, 'errors'):
            result.errors.extend(execution_results.errors or [])
        
        if hasattr(execution_results, 'metrics'):
            # Update metrics from execution
            result.metrics.tasks_completed = execution_results.metrics.get('tasks_completed', 0)
            result.metrics.tasks_failed = execution_results.metrics.get('tasks_failed', 0)
            result.metrics.files_modified = execution_results.metrics.get('files_modified', 0)
            result.metrics.lines_changed = execution_results.metrics.get('lines_changed', 0)
        
        # Set context data for debugging/learning
        result.context_data = {
            "context_confidence": context.confidence_score,
            "intent_category": intent_analysis.primary_intent.category.value,
            "intent_confidence": intent_analysis.primary_intent.confidence,
            "files_analyzed": len(context.relevant_files),
            "dependencies_found": len(context.dependencies) if context.dependencies else 0
        }
    
    async def get_engine_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics."""
        return {
            "engine_id": self.engine_id,
            "initialized": self.initialized,
            "total_commands_processed": self.total_commands_processed,
            "successful_commands": self.successful_commands,
            "success_rate": self.successful_commands / max(1, self.total_commands_processed),
            "average_processing_time_ms": self.average_processing_time_ms,
            "components_initialized": {
                "context_gatherer": self._context_gatherer is not None,
                "intent_analyzer": self._intent_analyzer is not None,
                "action_planner": self._action_planner is not None,
                "execution_router": self._execution_router is not None
            }
        }
    
    async def shutdown(self):
        """Shutdown the engine and cleanup resources."""
        logger.info(f"🔄 Shutting down UAE: {self.engine_id}")
        
        # Shutdown components in reverse order
        shutdown_tasks = []
        
        if self._execution_router:
            shutdown_tasks.append(self._execution_router.shutdown())
        if self._action_planner:
            shutdown_tasks.append(self._action_planner.shutdown())
        if self._intent_analyzer:
            shutdown_tasks.append(self._intent_analyzer.shutdown())
        if self._context_gatherer:
            shutdown_tasks.append(self._context_gatherer.shutdown())
        
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        self.initialized = False
        logger.info(f"✅ UAE shutdown complete: {self.engine_id}")