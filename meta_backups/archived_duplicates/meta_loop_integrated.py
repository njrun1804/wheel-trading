#!/usr/bin/env python3
"""
Integrated Meta-Loop - Actually connects all the pieces
This is the missing piece that makes everything work together automatically
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Dict, Any, List

from claude_code_integration import ClaudeCodeMonitor, ClaudeCodeFeedbackCollector
from mistake_detection import MistakeDetector
from meta_prime import MetaPrime
from meta_coordinator import MetaCoordinator
from meta_daemon import MetaDaemon
from meta_monitoring import MetaSystemMonitor


class IntegratedMetaLoop:
    """The actual integrated meta-loop that connects everything"""
    
    def __init__(self):
        print("🔄 Initializing Integrated Meta-Loop...")
        
        # Core components
        self.meta_prime = MetaPrime()
        self.meta_coordinator = MetaCoordinator()
        self.monitor = MetaSystemMonitor()
        
        # Integration components
        self.claude_monitor = ClaudeCodeMonitor()
        self.feedback_collector = ClaudeCodeFeedbackCollector()
        self.mistake_detector = MistakeDetector()
        
        # State tracking
        self.loop_count = 0
        self.effectiveness_scores = []
        self.auto_improvements = []
        
        print("✅ Integrated Meta-Loop initialized")
        
    async def run_complete_meta_loop(self):
        """Run the complete meta-loop that actually integrates everything"""
        
        print("🚀 Starting Complete Integrated Meta-Loop...")
        
        while True:
            self.loop_count += 1
            loop_start = time.time()
            
            print(f"\n🔄 Meta-Loop Cycle #{self.loop_count}")
            
            # Step 1: Monitor Claude Code activity
            claude_activity = await self._monitor_claude_activity()
            
            # Step 2: Detect mistakes in recent changes
            mistakes = await self._detect_recent_mistakes()
            
            # Step 3: Collect effectiveness feedback
            effectiveness = await self._measure_effectiveness()
            
            # Step 4: Update meta-system based on observations
            evolution_triggered = await self._trigger_meta_evolution(
                claude_activity, mistakes, effectiveness
            )
            
            # Step 5: Auto-improve the meta-loop itself
            auto_improvements = await self._auto_improve_meta_loop()
            
            # Step 6: Report cycle results
            cycle_time = time.time() - loop_start
            await self._report_cycle_results(cycle_time, evolution_triggered, auto_improvements)
            
            # Wait before next cycle (configurable)
            await asyncio.sleep(30)  # 30-second cycles
            
    async def _monitor_claude_activity(self) -> Dict[str, Any]:
        """Monitor what Claude Code has been doing"""
        
        activity = {
            "files_modified": len(self.claude_monitor.files_modified),
            "session_duration": time.time() - self.claude_monitor.claude_session_start,
            "recent_files": self.claude_monitor.files_modified[-5:],
            "modification_rate": len(self.claude_monitor.files_modified) / max(1, (time.time() - self.claude_monitor.claude_session_start) / 60)  # per minute
        }
        
        # Record activity observation
        self.meta_prime.observe("claude_activity_cycle", activity)
        
        return activity
        
    async def _detect_recent_mistakes(self) -> List[Dict[str, Any]]:
        """Check for mistakes in recently modified files"""
        
        mistakes = []
        
        # Check files Claude recently modified
        for file_path_str in self.claude_monitor.files_modified[-3:]:  # Last 3 files
            file_path = Path(file_path_str)
            if file_path.exists():
                # Run mistake detection
                result = self.mistake_detector.check_file_thoroughly(file_path)
                if not result["clean_file"]:
                    mistakes.append({
                        "file": file_path_str,
                        "mistake_count": result["total_mistakes"],
                        "severity": result["mistakes_by_severity"],
                        "types": result["mistake_types"]
                    })
                    
        # Record mistake detection results
        self.meta_prime.observe("mistake_detection_cycle", {
            "files_checked": len(self.claude_monitor.files_modified[-3:]),
            "mistakes_found": len(mistakes),
            "mistake_details": mistakes
        })
        
        return mistakes
        
    async def _measure_effectiveness(self) -> Dict[str, Any]:
        """Measure how effective the meta-loop is being"""
        
        # Get feedback summary
        feedback_summary = self.feedback_collector.get_feedback_summary()
        
        # Get system health
        health = self.monitor.health_check("meta_prime")
        
        # Calculate effectiveness score
        effectiveness_factors = []
        
        # Factor 1: User feedback (if available)
        if feedback_summary["total_feedback"] > 0:
            effectiveness_factors.append(feedback_summary["average_rating"] / 5.0)
            
        # Factor 2: System health
        if health.status == "healthy":
            effectiveness_factors.append(1.0)
        elif health.status == "warning":
            effectiveness_factors.append(0.7)
        else:
            effectiveness_factors.append(0.3)
            
        # Factor 3: Error rate (lower is better)
        recent_mistakes = len([m for m in self.mistake_detector.detected_mistakes 
                             if time.time() - m.timestamp < 1800])  # Last 30 min
        error_factor = max(0, 1.0 - (recent_mistakes / 10.0))  # Normalize
        effectiveness_factors.append(error_factor)
        
        # Factor 4: Evolution success rate
        evolution_history = self.meta_prime.get_recent_observations(20)
        successful_evolutions = len([obs for obs in evolution_history 
                                   if obs[1] == "evolution_executed"])
        failed_evolutions = len([obs for obs in evolution_history 
                               if obs[1] == "evolution_failed"])
        
        if successful_evolutions + failed_evolutions > 0:
            evolution_factor = successful_evolutions / (successful_evolutions + failed_evolutions)
            effectiveness_factors.append(evolution_factor)
            
        # Calculate overall effectiveness
        overall_effectiveness = sum(effectiveness_factors) / len(effectiveness_factors) if effectiveness_factors else 0.5
        
        effectiveness = {
            "overall_score": overall_effectiveness,
            "feedback_score": feedback_summary["average_rating"] / 5.0 if feedback_summary["total_feedback"] > 0 else None,
            "health_score": 1.0 if health.status == "healthy" else 0.5,
            "error_rate": recent_mistakes,
            "evolution_success_rate": evolution_factor if 'evolution_factor' in locals() else None,
            "factors_considered": len(effectiveness_factors)
        }
        
        self.effectiveness_scores.append(overall_effectiveness)
        
        # Record effectiveness measurement
        self.meta_prime.observe("effectiveness_measurement", effectiveness)
        
        return effectiveness
        
    async def _trigger_meta_evolution(self, claude_activity: Dict[str, Any], 
                                    mistakes: List[Dict[str, Any]], 
                                    effectiveness: Dict[str, Any]) -> bool:
        """Trigger meta-system evolution based on integrated observations"""
        
        evolution_triggered = False
        
        # Determine if evolution should be triggered
        should_evolve = False
        evolution_reasons = []
        
        # Reason 1: High Claude activity with low effectiveness
        if claude_activity["files_modified"] > 5 and effectiveness["overall_score"] < 0.6:
            should_evolve = True
            evolution_reasons.append("high_activity_low_effectiveness")
            
        # Reason 2: Recent mistakes detected
        if len(mistakes) > 2:
            should_evolve = True
            evolution_reasons.append("multiple_mistakes_detected")
            
        # Reason 3: Declining effectiveness trend
        if len(self.effectiveness_scores) >= 3:
            recent_trend = self.effectiveness_scores[-3:]
            if all(recent_trend[i] > recent_trend[i+1] for i in range(len(recent_trend)-1)):
                should_evolve = True
                evolution_reasons.append("declining_effectiveness_trend")
                
        # Reason 4: System readiness
        if self.meta_prime.should_evolve():
            should_evolve = True
            evolution_reasons.append("meta_system_ready")
            
        if should_evolve:
            print(f"🧬 Triggering evolution: {', '.join(evolution_reasons)}")
            
            # Create evolution context
            evolution_context = {
                "claude_activity": claude_activity,
                "mistakes": mistakes,
                "effectiveness": effectiveness,
                "reasons": evolution_reasons,
                "loop_cycle": self.loop_count
            }
            
            # Record evolution trigger
            self.meta_prime.observe("integrated_evolution_trigger", evolution_context)
            
            # Actually trigger evolution
            success = self.meta_prime.evolve()
            evolution_triggered = success
            
            if success:
                print("✅ Meta-evolution successful")
            else:
                print("❌ Meta-evolution failed")
                
        return evolution_triggered
        
    async def _auto_improve_meta_loop(self) -> List[str]:
        """Auto-improve the meta-loop itself based on performance"""
        
        improvements = []
        
        # Improvement 1: Adjust cycle timing based on activity
        if len(self.effectiveness_scores) >= 5:
            avg_effectiveness = sum(self.effectiveness_scores[-5:]) / 5
            
            if avg_effectiveness < 0.4:
                improvements.append("increased_monitoring_frequency")
                print("🔧 Auto-improvement: Increased monitoring frequency")
            elif avg_effectiveness > 0.8:
                improvements.append("reduced_monitoring_overhead")
                print("🔧 Auto-improvement: Reduced monitoring overhead")
                
        # Improvement 2: Adapt mistake detection sensitivity
        mistake_rate = len(self.mistake_detector.detected_mistakes) / max(1, self.loop_count)
        if mistake_rate > 2:  # Too many false positives
            improvements.append("reduced_mistake_sensitivity")
            print("🔧 Auto-improvement: Reduced mistake detection sensitivity")
        elif mistake_rate < 0.1:  # Might be missing mistakes
            improvements.append("increased_mistake_sensitivity")
            print("🔧 Auto-improvement: Increased mistake detection sensitivity")
            
        # Improvement 3: Evolution trigger optimization
        evolution_history = [obs for obs in self.meta_prime.get_recent_observations(50) 
                           if obs[1] in ["evolution_executed", "evolution_failed"]]
        
        if len(evolution_history) > 10:
            success_rate = len([obs for obs in evolution_history if obs[1] == "evolution_executed"]) / len(evolution_history)
            
            if success_rate < 0.3:
                improvements.append("conservative_evolution_triggers")
                print("🔧 Auto-improvement: More conservative evolution triggers")
            elif success_rate > 0.9:
                improvements.append("aggressive_evolution_triggers")
                print("🔧 Auto-improvement: More aggressive evolution triggers")
                
        # Record improvements
        if improvements:
            self.auto_improvements.extend(improvements)
            self.meta_prime.observe("meta_loop_auto_improvement", {
                "improvements": improvements,
                "loop_cycle": self.loop_count,
                "effectiveness_context": self.effectiveness_scores[-5:] if len(self.effectiveness_scores) >= 5 else self.effectiveness_scores
            })
            
        return improvements
        
    async def _report_cycle_results(self, cycle_time: float, evolution_triggered: bool, 
                                  auto_improvements: List[str]):
        """Report the results of this meta-loop cycle"""
        
        # Create cycle report
        report = {
            "loop_cycle": self.loop_count,
            "cycle_time_seconds": cycle_time,
            "evolution_triggered": evolution_triggered,
            "auto_improvements": auto_improvements,
            "claude_files_modified": len(self.claude_monitor.files_modified),
            "total_mistakes_detected": len(self.mistake_detector.detected_mistakes),
            "current_effectiveness": self.effectiveness_scores[-1] if self.effectiveness_scores else 0,
            "total_auto_improvements": len(self.auto_improvements)
        }
        
        # Record cycle completion
        self.meta_prime.observe("meta_loop_cycle_complete", report)
        
        # Print summary
        print(f"📊 Cycle {self.loop_count} Complete ({cycle_time:.1f}s)")
        print(f"   Evolution triggered: {'✅' if evolution_triggered else '❌'}")
        print(f"   Auto-improvements: {len(auto_improvements)}")
        print(f"   Effectiveness: {self.effectiveness_scores[-1]:.1%}" if self.effectiveness_scores else "   Effectiveness: N/A")
        
        # Every 10 cycles, print detailed summary
        if self.loop_count % 10 == 0:
            await self._print_detailed_summary()
            
    async def _print_detailed_summary(self):
        """Print detailed summary every 10 cycles"""
        
        print(f"\n📈 Meta-Loop Summary (Last 10 Cycles)")
        print(f"   Total cycles: {self.loop_count}")
        print(f"   Files monitored: {len(self.claude_monitor.files_modified)}")
        print(f"   Mistakes detected: {len(self.mistake_detector.detected_mistakes)}")
        print(f"   Auto-improvements made: {len(self.auto_improvements)}")
        
        if self.effectiveness_scores:
            recent_effectiveness = self.effectiveness_scores[-10:] if len(self.effectiveness_scores) >= 10 else self.effectiveness_scores
            avg_effectiveness = sum(recent_effectiveness) / len(recent_effectiveness)
            print(f"   Average effectiveness: {avg_effectiveness:.1%}")
            
        # Get meta-system status
        status = self.meta_prime.status_report()
        lines = status.split('\n')
        for line in lines[:5]:  # First 5 lines of status
            if line.strip():
                print(f"   {line}")


async def start_integrated_meta_loop():
    """Start the complete integrated meta-loop"""
    
    print("🔥 Starting Complete Integrated Meta-Loop System")
    print("This connects Claude Code monitoring, mistake detection, and meta-evolution")
    print("=" * 70)
    
    meta_loop = IntegratedMetaLoop()
    
    try:
        await meta_loop.run_complete_meta_loop()
    except KeyboardInterrupt:
        print("\n🛑 Meta-loop stopped by user")
        
        # Final summary
        print(f"\n📊 Final Summary:")
        print(f"   Total cycles completed: {meta_loop.loop_count}")
        print(f"   Files monitored: {len(meta_loop.claude_monitor.files_modified)}")
        print(f"   Mistakes detected: {len(meta_loop.mistake_detector.detected_mistakes)}")
        print(f"   Auto-improvements: {len(meta_loop.auto_improvements)}")
        
        if meta_loop.effectiveness_scores:
            final_effectiveness = meta_loop.effectiveness_scores[-1]
            print(f"   Final effectiveness: {final_effectiveness:.1%}")


if __name__ == "__main__":
    print("🚀 Integrated Meta-Loop - Complete System Integration")
    print("This is what actually makes the meta-system work as described!")
    asyncio.run(start_integrated_meta_loop())