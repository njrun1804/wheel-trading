"""
from __future__ import annotations

Intelligent MCP Request Router
Routes requests to optimal MCP based on capabilities, load, and performance history
"""

import asyncio
import time
import psutil
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MCPCapability:
    """Defines capabilities of an MCP server"""
    name: str
    capabilities: List[str]
    performance_score: float = 1.0
    resource_weight: float = 1.0  # How resource-intensive this MCP is
    specializations: Dict[str, float] = field(default_factory=dict)  # Task type -> efficiency

@dataclass
class MCPMetrics:
    """Runtime metrics for an MCP server"""
    name: str
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    active_requests: int = 0
    total_requests: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    last_update: float = field(default_factory=time.time)

@dataclass
class RoutingDecision:
    """Result of routing decision"""
    primary_mcp: str
    fallback_mcp: Optional[str]
    confidence: float
    reasoning: str
    estimated_latency_ms: float

class IntelligentRouter:
    """Routes requests to optimal MCP based on multiple factors"""
    
    def __init__(self):
        self.capabilities = self._initialize_capabilities()
        self.metrics: Dict[str, MCPMetrics] = {}
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.routing_history: deque = deque(maxlen=1000)
        self.load_threshold = 0.8  # 80% CPU/memory threshold
        
    def _initialize_capabilities(self) -> Dict[str, MCPCapability]:
        """Initialize MCP capabilities registry"""
        return {
            'filesystem': MCPCapability(
                name='filesystem',
                capabilities=['read', 'write', 'list', 'watch'],
                performance_score=0.9,
                resource_weight=0.5,
                specializations={
                    'file_operations': 1.0,
                    'directory_scan': 0.9,
                    'large_files': 0.8
                }
            ),
            'ripgrep': MCPCapability(
                name='ripgrep',
                capabilities=['search', 'pattern_match', 'regex'],
                performance_score=1.0,
                resource_weight=0.7,
                specializations={
                    'code_search': 1.0,
                    'pattern_search': 0.95,
                    'large_codebase': 0.9
                }
            ),
            'dependency-graph': MCPCapability(
                name='dependency-graph',
                capabilities=['imports', 'structure', 'symbols'],
                performance_score=0.95,
                resource_weight=0.6,
                specializations={
                    'symbol_lookup': 1.0,
                    'import_analysis': 0.95,
                    'code_structure': 0.9
                }
            ),
            'duckdb': MCPCapability(
                name='duckdb',
                capabilities=['query', 'analytics', 'aggregation'],
                performance_score=0.85,
                resource_weight=1.0,
                specializations={
                    'data_analysis': 1.0,
                    'time_series': 0.9,
                    'aggregations': 0.95
                }
            ),
            'python_analysis': MCPCapability(
                name='python_analysis',
                capabilities=['execute', 'analyze', 'compute'],
                performance_score=0.8,
                resource_weight=0.8,
                specializations={
                    'code_execution': 1.0,
                    'mathematical_computation': 0.9,
                    'data_processing': 0.85
                }
            ),
            'memory': MCPCapability(
                name='memory',
                capabilities=['store', 'retrieve', 'cache'],
                performance_score=0.95,
                resource_weight=0.3,
                specializations={
                    'context_storage': 1.0,
                    'cache_lookup': 0.95,
                    'state_management': 0.9
                }
            )
        }
    
    async def route_request(
        self, 
        task_type: str, 
        requirements: Dict[str, Any],
        context: Optional[Dict] = None
    ) -> RoutingDecision:
        """Route a request to the optimal MCP"""
        
        # Update metrics
        await self._update_metrics()
        
        # Find capable MCPs
        capable_mcps = self._find_capable_mcps(task_type, requirements)
        
        if not capable_mcps:
            return RoutingDecision(
                primary_mcp='filesystem',  # Default fallback
                fallback_mcp=None,
                confidence=0.3,
                reasoning="No specialized MCP found, using default",
                estimated_latency_ms=100
            )
        
        # Score each capable MCP
        scores = []
        for mcp in capable_mcps:
            score, factors = self._score_mcp(mcp, task_type, requirements, context)
            scores.append((mcp, score, factors))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select primary and fallback
        primary = scores[0][0]
        fallback = scores[1][0] if len(scores) > 1 else None
        confidence = scores[0][1]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(scores[0][2])
        
        # Estimate latency
        estimated_latency = self._estimate_latency(primary, task_type)
        
        # Record routing decision
        decision = RoutingDecision(
            primary_mcp=primary,
            fallback_mcp=fallback,
            confidence=confidence,
            reasoning=reasoning,
            estimated_latency_ms=estimated_latency
        )
        
        self._record_routing_decision(decision, task_type, requirements)
        
        return decision
    
    def _find_capable_mcps(self, task_type: str, requirements: Dict) -> List[str]:
        """Find MCPs capable of handling the task"""
        capable = []
        
        required_capabilities = requirements.get('capabilities', [])
        
        for name, cap in self.capabilities.items():
            # Check if MCP has required capabilities
            if all(req in cap.capabilities for req in required_capabilities):
                # Check if MCP specializes in this task type
                if task_type in cap.specializations and cap.specializations[task_type] > 0.5:
                    capable.append(name)
                elif not cap.specializations:  # General purpose MCP
                    capable.append(name)
        
        return capable
    
    def _score_mcp(
        self, 
        mcp_name: str, 
        task_type: str, 
        requirements: Dict,
        context: Optional[Dict]
    ) -> Tuple[float, Dict[str, float]]:
        """Score an MCP for a specific task"""
        
        cap = self.capabilities[mcp_name]
        metrics = self.metrics.get(mcp_name, MCPMetrics(name=mcp_name))
        
        factors = {
            'specialization': cap.specializations.get(task_type, 0.5),
            'performance': cap.performance_score,
            'availability': 1.0,
            'load': 1.0,
            'history': 0.5
        }
        
        # Adjust for current load
        if metrics.cpu_percent > 80:
            factors['load'] *= 0.5
        elif metrics.cpu_percent > 60:
            factors['load'] *= 0.8
            
        # Adjust for memory usage
        if metrics.memory_mb > 1000:  # Over 1GB
            factors['load'] *= 0.7
            
        # Adjust for active requests
        if metrics.active_requests > 5:
            factors['availability'] *= 0.6
        elif metrics.active_requests > 2:
            factors['availability'] *= 0.8
            
        # Consider historical performance
        if mcp_name in self.performance_history:
            recent = list(self.performance_history[mcp_name])[-10:]
            if recent:
                avg_success = sum(1 for r in recent if r['success']) / len(recent)
                avg_latency = sum(r['latency_ms'] for r in recent) / len(recent)
                
                factors['history'] = avg_success
                if avg_latency > 1000:
                    factors['performance'] *= 0.7
        
        # Apply requirements weights
        if 'priority' in requirements:
            if requirements['priority'] == 'latency':
                factors['performance'] *= 1.5
            elif requirements['priority'] == 'reliability':
                factors['history'] *= 1.5
        
        # Calculate weighted score
        weights = {
            'specialization': 0.3,
            'performance': 0.2,
            'availability': 0.2,
            'load': 0.2,
            'history': 0.1
        }
        
        score = sum(factors[k] * weights[k] for k in factors)
        
        return score, factors
    
    def _generate_reasoning(self, factors: Dict[str, float]) -> str:
        """Generate human-readable reasoning for routing decision"""
        reasons = []
        
        if factors['specialization'] > 0.8:
            reasons.append("highly specialized for this task")
        
        if factors['load'] < 0.8:
            reasons.append("currently under heavy load")
        elif factors['load'] > 0.9:
            reasons.append("low resource utilization")
            
        if factors['history'] > 0.9:
            reasons.append("excellent historical performance")
        elif factors['history'] < 0.5:
            reasons.append("recent reliability issues")
            
        if factors['availability'] < 0.7:
            reasons.append("handling multiple requests")
            
        return "Selected because: " + ", ".join(reasons) if reasons else "Default selection"
    
    def _estimate_latency(self, mcp_name: str, task_type: str) -> float:
        """Estimate latency based on historical data"""
        if mcp_name not in self.performance_history:
            # Default estimates
            base_latencies = {
                'memory': 10,
                'dependency-graph': 20,
                'filesystem': 50,
                'ripgrep': 100,
                'python_analysis': 200,
                'duckdb': 300
            }
            return base_latencies.get(mcp_name, 100)
        
        # Use historical average
        recent = list(self.performance_history[mcp_name])[-20:]
        similar_tasks = [r for r in recent if r.get('task_type') == task_type]
        
        if similar_tasks:
            return sum(r['latency_ms'] for r in similar_tasks) / len(similar_tasks)
        elif recent:
            return sum(r['latency_ms'] for r in recent) / len(recent)
        else:
            return 100
    
    async def _update_metrics(self):
        """Update runtime metrics for all MCPs"""
        # This would integrate with actual process monitoring
        # For now, using simulated data
        for name in self.capabilities:
            if name not in self.metrics:
                self.metrics[name] = MCPMetrics(name=name)
            
            # Simulate metrics update
            metrics = self.metrics[name]
            metrics.cpu_percent = psutil.cpu_percent() * 0.1  # Simulated
            metrics.memory_mb = psutil.virtual_memory().used / 1024 / 1024 * 0.01
            metrics.last_update = time.time()
    
    def _record_routing_decision(
        self, 
        decision: RoutingDecision, 
        task_type: str,
        requirements: Dict
    ):
        """Record routing decision for learning"""
        self.routing_history.append({
            'timestamp': time.time(),
            'task_type': task_type,
            'decision': decision,
            'requirements': requirements
        })
    
    def record_performance(
        self, 
        mcp_name: str, 
        task_type: str,
        latency_ms: float,
        success: bool,
        error: Optional[str] = None
    ):
        """Record actual performance for learning"""
        self.performance_history[mcp_name].append({
            'timestamp': time.time(),
            'task_type': task_type,
            'latency_ms': latency_ms,
            'success': success,
            'error': error
        })
        
        # Update metrics
        if mcp_name in self.metrics:
            metrics = self.metrics[mcp_name]
            metrics.total_requests += 1
            if not success:
                metrics.total_errors += 1
            
            # Update rolling averages
            history = list(self.performance_history[mcp_name])[-50:]
            if history:
                metrics.avg_latency_ms = sum(h['latency_ms'] for h in history) / len(history)
                metrics.success_rate = sum(1 for h in history if h['success']) / len(history)
    
    def get_routing_stats(self) -> Dict:
        """Get routing statistics"""
        stats = {
            'total_requests': len(self.routing_history),
            'mcp_usage': defaultdict(int),
            'task_distribution': defaultdict(int),
            'avg_confidence': 0.0,
            'mcp_performance': {}
        }
        
        if self.routing_history:
            for record in self.routing_history:
                stats['mcp_usage'][record['decision'].primary_mcp] += 1
                stats['task_distribution'][record['task_type']] += 1
            
            stats['avg_confidence'] = sum(
                r['decision'].confidence for r in self.routing_history
            ) / len(self.routing_history)
        
        for name, metrics in self.metrics.items():
            stats['mcp_performance'][name] = {
                'success_rate': metrics.success_rate,
                'avg_latency_ms': metrics.avg_latency_ms,
                'total_requests': metrics.total_requests
            }
        
        return dict(stats)
    
    def optimize_routing(self):
        """Optimize routing based on historical performance"""
        # Adjust capability scores based on actual performance
        for mcp_name, history in self.performance_history.items():
            if len(history) < 10:
                continue
                
            recent = list(history)[-50:]
            
            # Calculate actual performance score
            success_rate = sum(1 for r in recent if r['success']) / len(recent)
            avg_latency = sum(r['latency_ms'] for r in recent) / len(recent)
            
            # Update capability performance score
            if mcp_name in self.capabilities:
                cap = self.capabilities[mcp_name]
                
                # Weighted update
                cap.performance_score = (
                    cap.performance_score * 0.7 +
                    success_rate * 0.3
                )
                
                # Update specialization scores
                task_performance = defaultdict(list)
                for record in recent:
                    task_performance[record['task_type']].append(record)
                
                for task_type, records in task_performance.items():
                    task_success = sum(1 for r in records if r['success']) / len(records)
                    if task_type in cap.specializations:
                        cap.specializations[task_type] = (
                            cap.specializations[task_type] * 0.8 +
                            task_success * 0.2
                        )


# Example usage
async def demo():
    """Demonstrate intelligent routing"""
    router = IntelligentRouter()
    
    # Example routing decisions
    examples = [
        ("code_search", {"capabilities": ["search", "regex"], "priority": "latency"}),
        ("symbol_lookup", {"capabilities": ["symbols"], "priority": "latency"}),
        ("data_analysis", {"capabilities": ["query"], "priority": "reliability"}),
        ("file_operations", {"capabilities": ["read", "write"], "priority": "latency"})
    ]
    
    for task_type, requirements in examples:
        decision = await router.route_request(task_type, requirements)
        print(f"\nTask: {task_type}")
        print(f"  Primary MCP: {decision.primary_mcp}")
        print(f"  Fallback: {decision.fallback_mcp}")
        print(f"  Confidence: {decision.confidence:.2f}")
        print(f"  Reasoning: {decision.reasoning}")
        print(f"  Estimated latency: {decision.estimated_latency_ms:.0f}ms")
        
        # Simulate performance recording
        router.record_performance(
            decision.primary_mcp,
            task_type,
            decision.estimated_latency_ms * 1.1,
            True
        )
    
    # Show statistics
    print("\nRouting Statistics:")
    print(json.dumps(router.get_routing_stats(), indent=2))

if __name__ == "__main__":
    asyncio.run(demo())