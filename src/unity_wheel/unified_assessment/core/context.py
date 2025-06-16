"""
Context Gathering Layer - Einstein-powered semantic understanding and code analysis.

This module handles comprehensive context gathering by combining:
- Einstein semantic search for code understanding
- AST analysis for code structure
- Dependency graph navigation
- Historical pattern analysis
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FileContext:
    """Context information about a relevant file."""
    
    file_path: str
    content_snippet: str
    relevance_score: float
    file_type: str
    last_modified: float
    size_bytes: int
    line_count: int
    semantic_matches: List[str] = field(default_factory=list)
    ast_summary: Optional[Dict[str, Any]] = None


@dataclass 
class Pattern:
    """Code pattern identified in context analysis."""
    
    pattern_type: str  # function, class, import, usage, error_handling
    description: str
    examples: List[str]
    frequency: int
    confidence: float
    locations: List[str] = field(default_factory=list)


@dataclass
class DependencyGraph:
    """Dependency relationships in the codebase."""
    
    nodes: Dict[str, Dict[str, Any]]  # file/module -> metadata
    edges: List[Dict[str, Any]]  # dependencies
    cycles: List[List[str]]  # circular dependencies
    entry_points: List[str]
    depth_levels: Dict[str, int]


@dataclass
class HistoricalAction:
    """Previously executed action for learning."""
    
    command: str
    intent_category: str
    files_affected: List[str]
    success: bool
    duration_ms: float
    timestamp: float
    similarity_score: float = 0.0


@dataclass
class SystemState:
    """Current system state information."""
    
    git_branch: str
    uncommitted_changes: bool
    test_status: str
    build_status: str
    running_processes: List[str]
    resource_usage: Dict[str, float]
    last_deployment: Optional[float] = None


@dataclass
class ContextQuery:
    """Query parameters for context gathering."""
    
    command: str
    search_depth: int = 3
    include_dependencies: bool = True
    include_history: bool = True
    max_files: int = 50
    include_tests: bool = True
    optimization_target: str = "balanced"  # speed, accuracy, balanced


@dataclass
class GatheredContext:
    """Complete context gathered for a command."""
    
    query: ContextQuery
    relevant_files: List[FileContext]
    code_patterns: List[Pattern] 
    dependencies: Optional[DependencyGraph]
    historical_actions: List[HistoricalAction]
    system_state: SystemState
    confidence_score: float
    gathering_time_ms: float
    cache_hits: Dict[str, bool] = field(default_factory=dict)
    
    @property
    def file_paths(self) -> List[str]:
        """Get list of relevant file paths."""
        return [fc.file_path for fc in self.relevant_files]
    
    @property
    def primary_files(self) -> List[FileContext]:
        """Get files with highest relevance scores."""
        return sorted(self.relevant_files, key=lambda f: f.relevance_score, reverse=True)[:10]


@dataclass
class UnifiedContext:
    """Unified context that flows through the entire pipeline."""
    
    # Command Context
    original_command: str
    parsed_intent: Optional[Any] = None  # Intent object from intent analysis
    
    # Code Context (from Einstein)
    semantic_matches: List[Dict[str, Any]] = field(default_factory=list)
    code_structure: Optional[Dict[str, Any]] = None
    dependency_graph: Optional[DependencyGraph] = None
    
    # System Context
    available_tools: List[str] = field(default_factory=list)
    system_resources: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Historical Context
    similar_commands: List[HistoricalAction] = field(default_factory=list)
    previous_results: List[Dict[str, Any]] = field(default_factory=list)
    learned_patterns: List[Pattern] = field(default_factory=list)
    
    # Execution Context
    execution_plan: Optional[Any] = None  # ExecutionPlan from planning
    progress_state: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: List[Dict[str, Any]] = field(default_factory=list)


class ContextGatherer:
    """
    Context gathering component that orchestrates comprehensive context collection.
    
    Uses Einstein semantic search, AST analysis, dependency graphs, and historical
    data to build complete understanding of the codebase and command context.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Einstein integration (lazy loaded)
        self._einstein_index = None
        self._dependency_graph = None
        self._python_analyzer = None
        self._code_helper = None
        
        # Caching for performance
        self._file_cache = {}
        self._pattern_cache = {}
        self._dependency_cache = None
        
        # Performance tracking
        self.total_context_requests = 0
        self.average_gathering_time_ms = 0.0
        self.cache_hit_rate = 0.0
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize context gathering components."""
        if self.initialized:
            return
        
        start_time = time.perf_counter()  
        logger.info("🔍 Initializing Context Gatherer...")
        
        try:
            # Initialize Einstein components in parallel
            init_tasks = [
                self._init_einstein_index(),
                self._init_dependency_graph(),
                self._init_code_analyzers(),
                self._init_system_monitors()
            ]
            
            await asyncio.gather(*init_tasks, return_exceptions=True)
            
            self.initialized = True
            init_time_ms = (time.perf_counter() - start_time) * 1000
            
            logger.info(f"✅ Context Gatherer initialized in {init_time_ms:.1f}ms")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Context Gatherer: {e}")
            raise
    
    async def gather_context(
        self,
        command: str,
        hints: Optional[Dict[str, Any]] = None
    ) -> GatheredContext:
        """
        Gather comprehensive context for a command.
        
        Args:
            command: Natural language command
            hints: Optional hints to guide context gathering
            
        Returns:
            GatheredContext with all relevant information
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        # Create context query
        query = ContextQuery(
            command=command,
            search_depth=hints.get('search_depth', 3) if hints else 3,
            max_files=hints.get('max_files', 50) if hints else 50,
            optimization_target=hints.get('optimization_target', 'balanced') if hints else 'balanced'
        )
        
        logger.debug(f"🔍 Gathering context for: '{command[:60]}...'")
        
        try:
            # Parallel context gathering for optimal performance
            gathering_tasks = [
                self._gather_semantic_context(query),
                self._gather_structural_context(query),
                self._gather_dependency_context(query),
                self._gather_historical_context(query),
                self._gather_system_context(query)
            ]
            
            results = await asyncio.gather(*gathering_tasks, return_exceptions=True)
            
            # Process results
            semantic_files = results[0] if not isinstance(results[0], Exception) else []
            patterns = results[1] if not isinstance(results[1], Exception) else []
            dependencies = results[2] if not isinstance(results[2], Exception) else None
            historical = results[3] if not isinstance(results[3], Exception) else []
            system_state = results[4] if not isinstance(results[4], Exception) else SystemState(
                git_branch="unknown", uncommitted_changes=False, test_status="unknown",
                build_status="unknown", running_processes=[], resource_usage={}
            )
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(
                semantic_files, patterns, dependencies, historical
            )
            
            # Create gathered context
            gathering_time_ms = (time.perf_counter() - start_time) * 1000
            
            context = GatheredContext(
                query=query,
                relevant_files=semantic_files,
                code_patterns=patterns,
                dependencies=dependencies,
                historical_actions=historical,
                system_state=system_state,
                confidence_score=confidence,
                gathering_time_ms=gathering_time_ms
            )
            
            # Update statistics
            self.total_context_requests += 1
            self.average_gathering_time_ms = (
                (self.average_gathering_time_ms * (self.total_context_requests - 1) + gathering_time_ms)
                / self.total_context_requests
            )
            
            logger.debug(
                f"✅ Context gathered in {gathering_time_ms:.1f}ms "
                f"(confidence: {confidence:.2f}, files: {len(semantic_files)})"
            )
            
            return context
            
        except Exception as e:
            logger.error(f"❌ Context gathering failed: {e}")
            # Return minimal context on failure
            return GatheredContext(
                query=query,
                relevant_files=[],
                code_patterns=[],
                dependencies=None,
                historical_actions=[],
                system_state=SystemState(
                    git_branch="unknown", uncommitted_changes=False, test_status="unknown",
                    build_status="unknown", running_processes=[], resource_usage={}
                ),
                confidence_score=0.0,
                gathering_time_ms=(time.perf_counter() - start_time) * 1000
            )
    
    async def _init_einstein_index(self):
        """Initialize Einstein semantic search index."""
        try:
            # Lazy import to avoid circular dependencies
            from einstein.unified_index import get_unified_index
            self._einstein_index = await get_unified_index()
            logger.debug("✅ Einstein index initialized")
        except Exception as e:
            logger.warning(f"⚠️  Einstein index not available: {e}")
            self._einstein_index = None
    
    async def _init_dependency_graph(self):
        """Initialize dependency graph analyzer."""
        try:
            from src.unity_wheel.accelerated_tools.dependency_graph_turbo import get_dependency_graph
            self._dependency_graph = get_dependency_graph()
            logger.debug("✅ Dependency graph initialized")
        except Exception as e:
            logger.warning(f"⚠️  Dependency graph not available: {e}")
            self._dependency_graph = None
    
    async def _init_code_analyzers(self):
        """Initialize code analysis tools."""
        try:
            from src.unity_wheel.accelerated_tools.python_analysis_turbo import get_python_analyzer
            from src.unity_wheel.accelerated_tools.python_helpers_turbo import get_code_helper
            
            self._python_analyzer = get_python_analyzer()
            self._code_helper = get_code_helper()
            logger.debug("✅ Code analyzers initialized")
        except Exception as e:
            logger.warning(f"⚠️  Code analyzers not available: {e}")
            self._python_analyzer = None
            self._code_helper = None
    
    async def _init_system_monitors(self):
        """Initialize system monitoring."""
        # System monitoring is lightweight and always available
        logger.debug("✅ System monitors initialized")
    
    async def _gather_semantic_context(self, query: ContextQuery) -> List[FileContext]:
        """Gather semantic context using Einstein search."""
        if not self._einstein_index:
            return []
        
        try:
            # Search for semantically relevant files
            search_results = await self._einstein_index.search(
                query.command,
                max_results=query.max_files,
                include_content=True
            )
            
            file_contexts = []
            for result in search_results[:query.max_files]:
                file_path = result.get('file_path', '')
                if file_path:
                    context = FileContext(
                        file_path=file_path,
                        content_snippet=result.get('content', '')[:500],
                        relevance_score=result.get('score', 0.0),
                        file_type=Path(file_path).suffix.lstrip('.') or 'unknown',
                        last_modified=result.get('last_modified', 0),
                        size_bytes=result.get('size_bytes', 0),
                        line_count=result.get('line_count', 0),
                        semantic_matches=result.get('matches', [])
                    )
                    file_contexts.append(context)
            
            return file_contexts
            
        except Exception as e:
            logger.warning(f"⚠️  Semantic context gathering failed: {e}")
            return []
    
    async def _gather_structural_context(self, query: ContextQuery) -> List[Pattern]:
        """Gather structural context through code pattern analysis."""
        if not self._python_analyzer:
            return []
        
        try:
            # Analyze code patterns relevant to the command
            patterns = []
            
            # Extract key terms from command for pattern matching
            command_terms = query.command.lower().split()
            pattern_terms = [term for term in command_terms 
                           if len(term) > 3 and term not in ['the', 'and', 'for', 'with']]
            
            # Mock pattern analysis (would be implemented with real AST analysis)
            for term in pattern_terms:
                if 'auth' in term:
                    patterns.append(Pattern(
                        pattern_type="authentication",
                        description=f"Authentication patterns related to '{term}'",
                        examples=[],
                        frequency=1,
                        confidence=0.7
                    ))
                elif 'test' in term:
                    patterns.append(Pattern(
                        pattern_type="testing",
                        description=f"Testing patterns related to '{term}'",
                        examples=[],
                        frequency=1,
                        confidence=0.8
                    ))
                elif 'performance' in term or 'optimize' in term:
                    patterns.append(Pattern(
                        pattern_type="performance",
                        description=f"Performance patterns related to '{term}'",
                        examples=[],
                        frequency=1,
                        confidence=0.6
                    ))
            
            return patterns
            
        except Exception as e:
            logger.warning(f"⚠️  Structural context gathering failed: {e}")
            return []
    
    async def _gather_dependency_context(self, query: ContextQuery) -> Optional[DependencyGraph]:
        """Gather dependency context using dependency graph."""
        if not self._dependency_graph or not query.include_dependencies:
            return None
        
        try:
            # Build dependency graph for relevant files
            # This would use the actual dependency graph analyzer
            return DependencyGraph(
                nodes={},
                edges=[],
                cycles=[],
                entry_points=[],
                depth_levels={}
            )
            
        except Exception as e:
            logger.warning(f"⚠️  Dependency context gathering failed: {e}")
            return None
    
    async def _gather_historical_context(self, query: ContextQuery) -> List[HistoricalAction]:
        """Gather historical context from previous similar commands."""
        if not query.include_history:
            return []
        
        try:
            # Would query historical database for similar commands
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.warning(f"⚠️  Historical context gathering failed: {e}")
            return []
    
    async def _gather_system_context(self, query: ContextQuery) -> SystemState:
        """Gather current system state context."""
        try:
            import subprocess
            import psutil
            
            # Get git branch
            try:
                git_branch = subprocess.check_output(
                    ['git', 'branch', '--show-current'], 
                    text=True, 
                    timeout=5
                ).strip()
            except:
                git_branch = "unknown"
            
            # Check for uncommitted changes
            try:
                git_status = subprocess.check_output(
                    ['git', 'status', '--porcelain'], 
                    text=True, 
                    timeout=5
                )
                uncommitted_changes = len(git_status.strip()) > 0
            except:
                uncommitted_changes = False
            
            # Get resource usage
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
            except:
                cpu_percent = 0.0
                memory_percent = 0.0
            
            return SystemState(
                git_branch=git_branch,
                uncommitted_changes=uncommitted_changes,
                test_status="unknown",
                build_status="unknown", 
                running_processes=[],
                resource_usage={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent
                }
            )
            
        except Exception as e:
            logger.warning(f"⚠️  System context gathering failed: {e}")
            return SystemState(
                git_branch="unknown",
                uncommitted_changes=False,
                test_status="unknown",
                build_status="unknown",
                running_processes=[],
                resource_usage={}
            )
    
    def _calculate_confidence_score(
        self,
        files: List[FileContext],
        patterns: List[Pattern],
        dependencies: Optional[DependencyGraph],
        history: List[HistoricalAction]
    ) -> float:
        """Calculate overall confidence score for gathered context."""
        
        score = 0.0
        max_score = 0.0
        
        # File relevance contribution (40%)
        if files:
            avg_relevance = sum(f.relevance_score for f in files) / len(files)
            score += avg_relevance * 0.4
        max_score += 0.4
        
        # Pattern confidence contribution (30%)
        if patterns:
            avg_pattern_confidence = sum(p.confidence for p in patterns) / len(patterns)
            score += avg_pattern_confidence * 0.3
        max_score += 0.3
        
        # Dependency graph contribution (20%)
        if dependencies and dependencies.nodes:
            dependency_score = min(len(dependencies.nodes) / 10.0, 1.0)
            score += dependency_score * 0.2
        max_score += 0.2
        
        # Historical context contribution (10%)
        if history:
            historical_score = min(len(history) / 5.0, 1.0)
            score += historical_score * 0.1
        max_score += 0.1
        
        return score / max_score if max_score > 0 else 0.0
    
    async def shutdown(self):
        """Shutdown context gatherer and cleanup resources."""
        logger.debug("🔄 Shutting down Context Gatherer")
        
        # Clear caches
        self._file_cache.clear()
        self._pattern_cache.clear()
        self._dependency_cache = None
        
        self.initialized = False
        logger.debug("✅ Context Gatherer shutdown complete")