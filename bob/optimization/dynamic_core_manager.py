"""Dynamic core detection and load-aware task distribution for Bob.

This module automatically detects Mac hardware (M3, M4, M4 Pro, M4 Max) and
implements intelligent load balancing across performance cores.
"""

import asyncio
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import psutil

from ..utils.logging import get_component_logger


class MacChip(Enum):
    """Mac chip types with their core configurations."""
    M3 = "M3"
    M3_PRO = "M3_Pro" 
    M3_MAX = "M3_Max"
    M4 = "M4"
    M4_PRO = "M4_Pro"
    M4_MAX = "M4_Max"
    UNKNOWN = "Unknown"


@dataclass
class CoreConfiguration:
    """Core configuration for a Mac chip."""
    chip_type: MacChip
    p_cores: int
    e_cores: int
    total_cores: int
    unified_memory_gb: int
    gpu_cores: int
    
    # Performance characteristics
    p_core_base_ghz: float = 3.2
    e_core_base_ghz: float = 2.4
    efficiency_ratio: float = 0.7  # E-cores are ~70% as fast


@dataclass
class AgentLoad:
    """Real-time load tracking for an agent."""
    agent_id: str
    core_id: int
    current_tasks: int = 0
    total_completed: int = 0
    total_duration: float = 0.0
    avg_duration: float = 0.0
    cpu_utilization: float = 0.0
    last_update: float = field(default_factory=time.time)
    
    def update_load(self, task_duration: float):
        """Update load metrics after task completion."""
        self.current_tasks = max(0, self.current_tasks - 1)
        self.total_completed += 1
        self.total_duration += task_duration
        self.avg_duration = self.total_duration / self.total_completed
        self.last_update = time.time()
    
    def get_load_score(self) -> float:
        """Calculate current load score (lower = less loaded)."""
        # Base score from current tasks
        base_score = self.current_tasks * 10
        
        # Add penalty for high average duration
        duration_penalty = max(0, self.avg_duration - 1.0) * 5
        
        # Add penalty for high CPU utilization
        cpu_penalty = self.cpu_utilization * 2
        
        return base_score + duration_penalty + cpu_penalty


class DynamicCoreManager:
    """Manages dynamic core detection and load-aware task distribution."""
    
    def __init__(self):
        self.logger = get_component_logger("dynamic_core_manager")
        
        # Hardware configuration
        self.config: Optional[CoreConfiguration] = None
        self.p_core_agents: Dict[int, AgentLoad] = {}
        self.e_core_agents: Dict[int, AgentLoad] = {}
        
        # Load balancing settings
        self.use_e_cores = False  # Default: P-cores only
        self.e_core_spillover_threshold = 0.9  # Use E-cores when P-cores >90% loaded
        self.load_update_interval = 1.0  # Update load metrics every 1s
        
        # Performance monitoring
        self.performance_history: List[Dict] = []
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def initialize(self) -> None:
        """Initialize dynamic core management."""
        self.logger.info("🔍 Detecting Mac hardware configuration...")
        
        # Detect hardware
        self.config = await self._detect_mac_configuration()
        
        # Initialize agents based on configuration
        self._initialize_agents()
        
        # Start performance monitoring
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitor_performance())
        
        self.logger.info(f"✅ Dynamic core manager initialized for {self.config.chip_type.value}")
        self.logger.info(f"📊 Configuration: {self.config.p_cores}P + {self.config.e_cores}E cores, {self.config.unified_memory_gb}GB")
        self.logger.info(f"🎯 Strategy: {'P-cores only' if not self.use_e_cores else 'P-cores with E-core spillover'}")
    
    async def _detect_mac_configuration(self) -> CoreConfiguration:
        """Detect Mac chip type and core configuration."""
        try:
            # Get basic system info
            total_cores = psutil.cpu_count(logical=False)
            memory_bytes = psutil.virtual_memory().total
            memory_gb = memory_bytes // (1024 ** 3)
            
            # Try to detect P-cores using sysctl
            p_cores = 8  # Default fallback
            try:
                result = subprocess.run(['sysctl', '-n', 'hw.perflevel0.physicalcpu'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    p_cores = int(result.stdout.strip())
            except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
                self.logger.warning("Could not detect P-cores via sysctl, using default")
            
            e_cores = total_cores - p_cores
            
            # Try to detect chip type from system_profiler
            chip_type = MacChip.UNKNOWN
            gpu_cores = 20  # Default
            
            try:
                result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    output = result.stdout.lower()
                    if 'm4 max' in output:
                        chip_type = MacChip.M4_MAX
                        gpu_cores = 40
                    elif 'm4 pro' in output:
                        chip_type = MacChip.M4_PRO  
                        gpu_cores = 20
                    elif 'm4' in output:
                        chip_type = MacChip.M4
                        gpu_cores = 10
                    elif 'm3 max' in output:
                        chip_type = MacChip.M3_MAX
                        gpu_cores = 40
                    elif 'm3 pro' in output:
                        chip_type = MacChip.M3_PRO
                        gpu_cores = 18
                    elif 'm3' in output:
                        chip_type = MacChip.M3
                        gpu_cores = 10
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.logger.warning("Could not detect chip type, using heuristics")
                
                # Heuristic detection based on core count
                if p_cores >= 12 and total_cores >= 16:
                    chip_type = MacChip.M4_MAX
                    gpu_cores = 40
                elif p_cores >= 8 and total_cores >= 12:
                    chip_type = MacChip.M4_PRO
                    gpu_cores = 20
                elif p_cores >= 4 and total_cores >= 8:
                    chip_type = MacChip.M4
                    gpu_cores = 10
                
                # Special handling for M4 Max Studio with 128GB RAM
                if memory_gb >= 120 and p_cores >= 12:
                    chip_type = MacChip.M4_MAX
                    gpu_cores = 40
                    self.logger.info("🏢 Detected M4 Max Studio configuration!")
            
            config = CoreConfiguration(
                chip_type=chip_type,
                p_cores=p_cores,
                e_cores=e_cores,
                total_cores=total_cores,
                unified_memory_gb=memory_gb,
                gpu_cores=gpu_cores
            )
            
            self.logger.info(f"🖥️  Detected: {chip_type.value}")
            self.logger.info(f"🧮 Cores: {p_cores} P-cores + {e_cores} E-cores")
            self.logger.info(f"💾 Memory: {memory_gb}GB unified")
            self.logger.info(f"🎮 GPU: {gpu_cores} cores")
            
            return config
            
        except Exception as e:
            self.logger.error(f"Hardware detection failed: {e}")
            # Fallback configuration
            return CoreConfiguration(
                chip_type=MacChip.UNKNOWN,
                p_cores=8,
                e_cores=4, 
                total_cores=12,
                unified_memory_gb=24,
                gpu_cores=20
            )
    
    def _initialize_agents(self) -> None:
        """Initialize agent load tracking."""
        # Initialize P-core agents
        for i in range(self.config.p_cores):
            agent_id = f"p_core_agent_{i}"
            self.p_core_agents[i] = AgentLoad(agent_id=agent_id, core_id=i)
        
        # Initialize E-core agents (even if not used initially)
        for i in range(self.config.e_cores):
            agent_id = f"e_core_agent_{i}"
            self.e_core_agents[i] = AgentLoad(agent_id=agent_id, core_id=i + self.config.p_cores)
        
        self.logger.info(f"🤖 Initialized {len(self.p_core_agents)} P-core agents")
        if self.use_e_cores:
            self.logger.info(f"🤖 Initialized {len(self.e_core_agents)} E-core agents")
    
    def select_optimal_agent(self, task_requirements: Dict) -> Tuple[str, bool]:
        """Select optimal agent for task based on current load.
        
        Returns:
            Tuple of (agent_id, is_p_core)
        """
        cpu_intensive = task_requirements.get('cpu_intensive', True)
        estimated_duration = task_requirements.get('estimated_duration', 1.0)
        
        # For CPU-intensive tasks, prefer P-cores
        if cpu_intensive:
            # Check P-core availability
            best_p_core = self._find_least_loaded_agent(self.p_core_agents)
            
            if best_p_core is not None:
                agent = self.p_core_agents[best_p_core]
                # Use P-core if not overloaded
                if agent.get_load_score() < 50:  # Reasonable load threshold
                    agent.current_tasks += 1
                    return agent.agent_id, True
            
            # P-cores overloaded - consider E-core spillover
            if self.use_e_cores and self._should_use_e_core_spillover():
                best_e_core = self._find_least_loaded_agent(self.e_core_agents)
                if best_e_core is not None:
                    agent = self.e_core_agents[best_e_core]
                    agent.current_tasks += 1
                    self.logger.debug(f"🔄 Spilling CPU task to E-core due to P-core load")
                    return agent.agent_id, False
            
            # Fallback to least loaded P-core even if overloaded
            if best_p_core is not None:
                agent = self.p_core_agents[best_p_core]
                agent.current_tasks += 1
                return agent.agent_id, True
        
        # For I/O tasks, can use E-cores if enabled
        else:
            if self.use_e_cores:
                best_e_core = self._find_least_loaded_agent(self.e_core_agents)
                if best_e_core is not None:
                    agent = self.e_core_agents[best_e_core]
                    agent.current_tasks += 1
                    return agent.agent_id, False
            
            # Fallback to P-core for I/O tasks
            best_p_core = self._find_least_loaded_agent(self.p_core_agents)
            if best_p_core is not None:
                agent = self.p_core_agents[best_p_core]
                agent.current_tasks += 1
                return agent.agent_id, True
        
        # Ultimate fallback
        agent = self.p_core_agents[0]
        agent.current_tasks += 1
        return agent.agent_id, True
    
    def _find_least_loaded_agent(self, agents: Dict[int, AgentLoad]) -> Optional[int]:
        """Find the least loaded agent in a group."""
        if not agents:
            return None
        
        return min(agents.keys(), key=lambda i: agents[i].get_load_score())
    
    def _should_use_e_core_spillover(self) -> bool:
        """Check if we should use E-core spillover."""
        if not self.use_e_cores:
            return False
        
        # Calculate average P-core load
        if not self.p_core_agents:
            return False
        
        total_p_core_load = sum(agent.current_tasks for agent in self.p_core_agents.values())
        avg_p_core_load = total_p_core_load / len(self.p_core_agents)
        
        # Use spillover if average load exceeds threshold
        return avg_p_core_load > self.e_core_spillover_threshold
    
    def task_completed(self, agent_id: str, duration: float) -> None:
        """Update load tracking when task completes."""
        # Find the agent
        agent = None
        for p_agent in self.p_core_agents.values():
            if p_agent.agent_id == agent_id:
                agent = p_agent
                break
        
        if agent is None:
            for e_agent in self.e_core_agents.values():
                if e_agent.agent_id == agent_id:
                    agent = e_agent
                    break
        
        if agent:
            agent.update_load(duration)
    
    async def _monitor_performance(self) -> None:
        """Monitor system performance and adjust strategy."""
        while self._running:
            try:
                await asyncio.sleep(self.load_update_interval)
                
                # Update CPU utilization for all agents
                per_core_cpu = psutil.cpu_percent(percpu=True, interval=0.1)
                
                # Update P-core agents
                for i, agent in self.p_core_agents.items():
                    if i < len(per_core_cpu):
                        agent.cpu_utilization = per_core_cpu[i] / 100.0
                
                # Update E-core agents
                for i, agent in self.e_core_agents.items():
                    core_idx = i + self.config.p_cores
                    if core_idx < len(per_core_cpu):
                        agent.cpu_utilization = per_core_cpu[core_idx] / 100.0
                
                # Record performance snapshot
                self._record_performance_snapshot()
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(5.0)
    
    def _record_performance_snapshot(self) -> None:
        """Record current performance state."""
        snapshot = {
            'timestamp': time.time(),
            'p_core_utilization': sum(a.cpu_utilization for a in self.p_core_agents.values()) / len(self.p_core_agents),
            'e_core_utilization': sum(a.cpu_utilization for a in self.e_core_agents.values()) / len(self.e_core_agents) if self.e_core_agents else 0,
            'p_core_load': sum(a.current_tasks for a in self.p_core_agents.values()),
            'e_core_load': sum(a.current_tasks for a in self.e_core_agents.values()),
            'total_tasks_completed': sum(a.total_completed for a in self.p_core_agents.values()) + sum(a.total_completed for a in self.e_core_agents.values())
        }
        
        # Keep last 100 snapshots
        self.performance_history.append(snapshot)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report."""
        if not self.performance_history:
            return {"error": "No performance data available"}
        
        latest = self.performance_history[-1]
        
        # Calculate load distribution evenness
        p_core_loads = [agent.current_tasks for agent in self.p_core_agents.values()]
        load_variance = sum((load - sum(p_core_loads)/len(p_core_loads))**2 for load in p_core_loads) / len(p_core_loads)
        evenness_score = 1.0 / (1.0 + load_variance)  # Higher = more even
        
        return {
            'hardware_config': {
                'chip_type': self.config.chip_type.value,
                'p_cores': self.config.p_cores,
                'e_cores': self.config.e_cores,
                'unified_memory_gb': self.config.unified_memory_gb,
                'gpu_cores': self.config.gpu_cores
            },
            'current_performance': latest,
            'load_distribution': {
                'p_core_loads': p_core_loads,
                'evenness_score': evenness_score,
                'load_variance': load_variance
            },
            'strategy': {
                'use_e_cores': self.use_e_cores,
                'spillover_threshold': self.e_core_spillover_threshold
            },
            'agent_details': {
                'p_core_agents': [
                    {
                        'agent_id': agent.agent_id,
                        'current_tasks': agent.current_tasks,
                        'total_completed': agent.total_completed,
                        'avg_duration': agent.avg_duration,
                        'cpu_utilization': agent.cpu_utilization,
                        'load_score': agent.get_load_score()
                    }
                    for agent in self.p_core_agents.values()
                ],
                'e_core_agents': [
                    {
                        'agent_id': agent.agent_id,
                        'current_tasks': agent.current_tasks,
                        'total_completed': agent.total_completed,
                        'avg_duration': agent.avg_duration,
                        'cpu_utilization': agent.cpu_utilization,
                        'load_score': agent.get_load_score()
                    }
                    for agent in self.e_core_agents.values()
                ] if self.use_e_cores else []
            }
        }
    
    def enable_e_core_spillover(self, threshold: float = 0.9) -> None:
        """Enable E-core spillover when P-cores are overloaded."""
        self.use_e_cores = True
        self.e_core_spillover_threshold = threshold
        self.logger.info(f"✅ Enabled E-core spillover at {threshold:.1%} P-core load")
    
    def disable_e_cores(self) -> None:
        """Disable E-core usage - P-cores only."""
        self.use_e_cores = False
        self.logger.info("✅ Disabled E-core usage - P-cores only mode")
    
    async def shutdown(self) -> None:
        """Shutdown the core manager."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("🛑 Dynamic core manager shutdown complete")


# Global instance
_global_core_manager: Optional[DynamicCoreManager] = None


async def get_core_manager() -> DynamicCoreManager:
    """Get or create global core manager instance."""
    global _global_core_manager
    if _global_core_manager is None:
        _global_core_manager = DynamicCoreManager()
        await _global_core_manager.initialize()
    return _global_core_manager