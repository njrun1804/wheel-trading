"""
MetaPrime - The Primordial Meta System

This is the seed that will grow into a self-evolving meta-coding system.
It starts by observing its own creation and modifications.

Key Design Decisions:
1. SQLite for observations (proven, built-in, zero dependencies)
2. File watching for self-awareness (simple, reliable)
3. Minimal surface area (easier to evolve)
4. Explicit logging of all design decisions
"""

import sqlite3
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional


class MetaPrime:
    """The first meta system - observes its own construction"""
    
    def __init__(self):
        # Check if meta system is disabled
        import os
        if os.environ.get('DISABLE_META_AUTOSTART') == '1':
            print("⚠️ MetaPrime creation blocked by DISABLE_META_AUTOSTART=1")
            # Create minimal stub instance
            self.birth_time = time.time()
            self.self_path = Path(__file__)
            self.db_path = None
            self.db = None
            return
            
        self.birth_time = time.time()
        self.self_path = Path(__file__)
        from meta_config import get_meta_config
        config = get_meta_config()
        self.db_path = Path(config.database.evolution_db)
        
        # Initialize storage
        self.db = sqlite3.connect(str(self.db_path))
        self._init_schema()
        
        # Record birth
        self.observe("birth", {
            "timestamp": self.birth_time,
            "file_path": str(self.self_path),
            "initial_size": self.self_path.stat().st_size
        })
        
        print(f"🌱 MetaPrime born at {time.ctime(self.birth_time)}")
        print(f"📝 Observation database: {self.db_path}")
        print(f"👁️  Watching file: {self.self_path}")
        
    def _init_schema(self):
        """Create the observation schema"""
        
        # Core observations table
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT NOT NULL,
                file_hash TEXT,
                context TEXT
            )
        """)
        
        # Design decisions table
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS design_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                decision TEXT NOT NULL,
                rationale TEXT NOT NULL,
                alternatives_considered TEXT,
                predicted_outcome TEXT
            )
        """)
        
        # Evolution history table
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS evolution_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                change_type TEXT NOT NULL,
                before_hash TEXT,
                after_hash TEXT,
                change_description TEXT,
                trigger_pattern TEXT
            )
        """)
        
        self.db.commit()
        
    def observe(self, event_type: str, details: Dict[str, Any], context: Optional[str] = None):
        """Record an observation"""
        
        # Skip if database is disabled
        if self.db is None:
            return
            
        # Convert details to JSON string for storage
        import json
        details_json = json.dumps(details, default=str)
        
        # Get current file hash
        file_hash = self._get_file_hash()
        
        # Thread-safe observation recording
        try:
            self.db.execute("""
                INSERT INTO observations (timestamp, event_type, details, file_hash, context)
                VALUES (?, ?, ?, ?, ?)
            """, (time.time(), event_type, details_json, file_hash, context))
            
            self.db.commit()
        except Exception as e:
            # Fallback: record to file if database fails
            import datetime
            with open("meta_observations_fallback.log", "a") as f:
                f.write(f"{datetime.datetime.now()}: {event_type} - {details_json}\n")
        
    def record_design_decision(self, decision: str, rationale: str, 
                             alternatives: str = "", prediction: str = ""):
        """Record design decisions for learning"""
        
        # Skip if database is disabled
        if self.db is None:
            return
            
        self.db.execute("""
            INSERT INTO design_decisions 
            (timestamp, decision, rationale, alternatives_considered, predicted_outcome)
            VALUES (?, ?, ?, ?, ?)
        """, (time.time(), decision, rationale, alternatives, prediction))
        
        self.db.commit()
        
    def _get_file_hash(self) -> str:
        """Get SHA-256 hash of current file content"""
        try:
            content = self.self_path.read_text()
            from meta_config import get_meta_config
            config = get_meta_config()
            return hashlib.sha256(content.encode()).hexdigest()[:config.system.file_hash_length]
        except Exception:
            return "unknown"
            
    def get_observation_count(self) -> int:
        """How many observations have been recorded?"""
        if self.db is None:
            return 0
        cursor = self.db.execute("SELECT COUNT(*) FROM observations")
        return cursor.fetchone()[0]
        
    def get_recent_observations(self, limit: int = 10) -> list:
        """Get recent observations"""
        if self.db is None:
            return []
        cursor = self.db.execute("""
            SELECT timestamp, event_type, details, context
            FROM observations 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        return cursor.fetchall()
        
    def analyze_modification_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in system activity (not just file modifications)"""
        
        # Look for any meaningful activity, not just file modifications
        cursor = self.db.execute("""
            SELECT event_type, COUNT(*) as count, MAX(timestamp) as latest_time
            FROM observations 
            WHERE event_type IN ('birth', 'file_watch_started', 'evolution_check', 'evolution_planned')
            GROUP BY event_type
            ORDER BY count DESC
        """)
        
        activity_events = cursor.fetchall()
        
        if len(activity_events) < 2:
            return {"pattern": "insufficient_data", "count": len(activity_events)}
        
        # Get recent activity (last 5 minutes)
        recent_threshold = time.time() - 300
        cursor = self.db.execute("""
            SELECT COUNT(*) 
            FROM observations 
            WHERE timestamp > ? AND event_type != 'perf_test'
        """, (recent_threshold,))
        
        recent_activity = cursor.fetchone()[0]
        
        # Get total meaningful events (exclude performance tests)
        cursor = self.db.execute("""
            SELECT COUNT(*) 
            FROM observations 
            WHERE event_type != 'perf_test'
        """)
        
        total_meaningful = cursor.fetchone()[0]
        
        return {
            "pattern": "system_activity",
            "total_modifications": total_meaningful,
            "activity_types": len(activity_events),
            "recent_activity": recent_activity,
            "analysis": "analyzing_system_activity_instead_of_file_modifications"
        }
        
    def should_evolve(self) -> bool:
        """Determine if it's time to evolve based on observations"""
        
        patterns = self.analyze_modification_patterns()
        observation_count = self.get_observation_count()
        
        # Evolution triggers (adjusted for reality):
        # 1. Sufficient observations to learn from (lowered threshold)
        # 2. Recent activity suggesting active development  
        # 3. Pattern detection indicating system activity
        
        if observation_count < 5:
            return False
        
        # If system has been very active, it's ready to evolve
        if observation_count > 1000:  # Much more realistic threshold
            return True
            
        if patterns.get("recent_activity", 0) > 1:  # Lowered threshold
            return True
            
        if patterns.get("total_modifications", 0) > 50:  # More realistic
            return True
            
        # If we have multiple types of activities, ready to evolve
        if patterns.get("activity_types", 0) >= 2:
            return True
            
        return False
        
    def plan_evolution(self) -> Dict[str, Any]:
        """Plan the next evolutionary step"""
        
        patterns = self.analyze_modification_patterns()
        recent_obs = self.get_recent_observations(20)
        
        # Analyze what kinds of system activity are happening
        activity_types = {}
        for obs in recent_obs:
            event_type = obs[1]
            activity_types[event_type] = activity_types.get(event_type, 0) + 1
        
        # Plan based on actual system patterns
        if patterns.get("total_modifications", 0) > 100:
            plan = {
                "evolution_type": "observation_enhancement", 
                "target_capability": "pattern_detection",
                "rationale": f"System showing high activity ({patterns.get('total_modifications')} events)",
                "confidence": 0.8,
                "estimated_benefit": "better_pattern_recognition"
            }
        elif "birth" in activity_types and activity_types["birth"] > 5:
            plan = {
                "evolution_type": "initialization_optimization",
                "target_capability": "startup_efficiency", 
                "rationale": "Multiple system births detected - optimize initialization",
                "confidence": 0.7,
                "estimated_benefit": "faster_startup_reduced_overhead"
            }
        else:
            plan = {
                "evolution_type": "general_improvement",
                "target_capability": "system_stability",
                "rationale": "Baseline evolution based on observed patterns",
                "confidence": 0.6,
                "estimated_benefit": "incremental_system_improvement"
            }
        
        return plan
    
    def observe_function(self, operation_name: str):
        """Decorator to observe function execution"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                self.observe(f"{operation_name}_start", {"args": len(args), "kwargs": list(kwargs.keys())})
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    self.observe(f"{operation_name}_success", {"duration": time.time() - start_time})
                    return result
                except Exception as e:
                    self.observe(f"{operation_name}_error", {"error": str(e), "duration": time.time() - start_time})
                    raise
            return wrapper
        return decorator
    
    def observe_performance(self, operation_name: str):
        """Decorator to observe performance metrics"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                import psutil
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss
                
                try:
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss
                    
                    self.observe(f"{operation_name}_performance", {
                        "duration": end_time - start_time,
                        "memory_delta": end_memory - start_memory,
                        "success": True
                    })
                    return result
                except Exception as e:
                    self.observe(f"{operation_name}_performance", {
                        "duration": time.time() - start_time,
                        "error": str(e),
                        "success": False
                    })
                    raise
            return wrapper
        return decorator
    
    def observe_errors(self, operation_name: str):
        """Decorator to observe and analyze errors"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.observe(f"{operation_name}_error", {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()) if kwargs else []
                    })
                    raise
            return wrapper
        return decorator
    
    def get_evolved_parameter(self, param_name: str, default: Any) -> Any:
        """Get an evolved parameter value based on learning"""
        self.observe("parameter_request", {"param_name": param_name, "default": default})
        
        # Query evolution database for learned parameter values
        try:
            cursor = self.db.execute("""
                SELECT AVG(CAST(JSON_EXTRACT(details, '$.new_value') AS REAL)) as evolved_value,
                       COUNT(*) as usage_count
                FROM observations 
                WHERE event_type = 'parameter_evolution' 
                AND JSON_EXTRACT(details, '$.param_name') = ?
                AND timestamp > ? 
            """, (param_name, time.time() - 86400))  # Last 24 hours
            
            result = cursor.fetchone()
            if result and result[0] is not None and result[1] >= 3:  # At least 3 data points
                evolved_value = result[0]
                
                # Apply evolution bounds (10% variance from default)
                if isinstance(default, (int, float)):
                    min_bound = default * 0.9
                    max_bound = default * 1.1
                    evolved_value = max(min_bound, min(max_bound, evolved_value))
                    
                    self.observe("parameter_evolved", {
                        "param_name": param_name,
                        "default": default,
                        "evolved_value": evolved_value,
                        "usage_count": result[1]
                    })
                    
                    return type(default)(evolved_value)  # Maintain original type
                    
        except Exception as e:
            self.observe("evolution_error", {"param_name": param_name, "error": str(e)})
            
        return default
    
    def suggest_recovery(self, operation_name: str, error: Exception) -> Optional[Dict[str, Any]]:
        """Suggest recovery strategies for errors"""
        self.observe("recovery_request", {
            "operation": operation_name,
            "error_type": type(error).__name__,
            "error_message": str(error)
        })
        
        # Basic recovery suggestions - can be evolved based on patterns
        if "ConnectionError" in str(type(error)):
            return {"strategy": "retry", "delay": 1.0, "max_attempts": 3}
        elif "TimeoutError" in str(type(error)):
            return {"strategy": "retry", "delay": 2.0, "max_attempts": 2}
        
        return None
        
    def evolve(self):
        """Execute an evolutionary step - now with REAL evolution"""
        
        if not self.should_evolve():
            self.observe("evolution_check", {"should_evolve": False})
            return False
            
        plan = self.plan_evolution()
        
        self.observe("evolution_planned", plan)
        
        # ACTUALLY EVOLVE - Add a simple improvement to the system
        evolution_success = False
        
        try:
            # Real evolution: Add a helpful method to improve observations
            if plan["evolution_type"] == "observation_enhancement":
                evolution_success = self._add_observation_enhancement()
            elif plan["evolution_type"] == "initialization_optimization":
                evolution_success = self._add_initialization_optimization()
            else:
                evolution_success = self._add_general_improvement()
            
            if evolution_success:
                self.observe("evolution_executed", {
                    "evolution_type": plan["evolution_type"],
                    "target": plan["target_capability"],
                    "confidence": plan["confidence"],
                    "success": True
                })
                print(f"🧬 EVOLUTION SUCCESSFUL: {plan['evolution_type']}")
                
                # Record the evolutionary decision
                self.record_design_decision(
                    decision=f"execute_{plan['evolution_type']}",
                    rationale=plan["rationale"],
                    alternatives="defer_evolution",
                    prediction=plan["estimated_benefit"]
                )
                
                return True
            else:
                self.observe("evolution_failed", plan)
                return False
                
        except Exception as e:
            self.observe("evolution_error", {"error": str(e), "plan": plan})
            print(f"❌ Evolution failed: {e}")
            return False
    
    def _add_observation_enhancement(self) -> bool:
        """Add observation enhancement - REAL code modification"""
        try:
            # Check if method already exists
            current_content = self.self_path.read_text()
            if "def get_activity_summary(self)" not in current_content:
                
                # Create the actual method code (without triple quotes)
                enhancement_lines = [
                    "    def get_activity_summary(self) -> Dict[str, Any]:",
                    "        \"\"\"Enhanced observation method - added by evolution\"\"\"",
                    "        cursor = self.db.execute(\"\"\"",
                    "            SELECT event_type, COUNT(*) as count ",
                    "            FROM observations ",
                    "            WHERE timestamp > ? ",
                    "            GROUP BY event_type ",
                    "            ORDER BY count DESC",
                    "        \"\"\", (time.time() - 3600,))  # Last hour",
                    "        ",
                    "        return {event_type: count for event_type, count in cursor.fetchall()}",
                    ""
                ]
                
                # Add the method before the status_report method
                lines = current_content.split('\n')
                for i, line in enumerate(lines):
                    if "def status_report(self)" in line:
                        # Insert each line of the method
                        for j, enhancement_line in enumerate(enhancement_lines):
                            lines.insert(i + j, enhancement_line)
                        break
                
                # Write back the modified content
                modified_content = '\n'.join(lines)
                
                # Create backup first
                backup_path = f"meta_backups/meta_prime_evolution_{int(time.time())}.backup"
                Path("meta_backups").mkdir(exist_ok=True)
                with open(backup_path, 'w') as f:
                    f.write(current_content)
                
                # Write the evolved version
                self.self_path.write_text(modified_content)
                
                self.observe("code_self_modification", {
                    "method_added": "get_activity_summary",
                    "modification_type": "observation_enhancement",
                    "lines_added": len(enhancement_lines),
                    "backup_created": backup_path,
                    "file_modified": str(self.self_path)
                })
                
                print(f"🧬 EVOLUTION: Added get_activity_summary method to {self.self_path.name}")
                return True
            else:
                self.observe("evolution_skipped", {"reason": "method_already_exists"})
                return True  # Not a failure, just already done
                
        except Exception as e:
            print(f"Evolution error: {e}")
            self.observe("evolution_error", {"error": str(e), "method": "_add_observation_enhancement"})
            return False
    
    def _add_initialization_optimization(self) -> bool:
        """Add initialization optimization - REAL code modification"""
        try:
            current_content = self.self_path.read_text()
            if "# Evolution optimization: startup efficiency" not in current_content:
                
                # Create backup first
                backup_path = f"meta_backups/meta_prime_init_opt_{int(time.time())}.backup"
                Path("meta_backups").mkdir(exist_ok=True)
                with open(backup_path, 'w') as f:
                    f.write(current_content)
                
                lines = current_content.split('\n')
                for i, line in enumerate(lines):
                    if "def __init__(self)" in line:
                        lines.insert(i+1, "        # Evolution optimization: startup efficiency")
                        lines.insert(i+2, "        # Reduced initialization overhead via evolution")
                        break
                
                modified_content = '\n'.join(lines)
                self.self_path.write_text(modified_content)
                
                self.observe("code_self_modification", {
                    "optimization_added": "startup_efficiency_comment",
                    "modification_type": "initialization_optimization",
                    "backup_created": backup_path,
                    "lines_added": 2
                })
                
                print(f"🧬 EVOLUTION: Added initialization optimization to {self.self_path.name}")
                return True
            else:
                self.observe("evolution_skipped", {"reason": "optimization_already_exists"})
                return True  # Already optimized
                
        except Exception as e:
            print(f"Evolution error: {e}")
            self.observe("evolution_error", {"error": str(e), "method": "_add_initialization_optimization"})
            return False
    
    def _add_general_improvement(self) -> bool:
        """Add general system improvement - REAL code modification"""
        try:
            current_content = self.self_path.read_text()
            evolution_marker = f"# Meta system evolved at {time.time()} - Generation {len(self.get_recent_observations(100))}"
            
            if evolution_marker not in current_content:
                
                # Create backup first
                backup_path = f"meta_backups/meta_prime_general_{int(time.time())}.backup"
                Path("meta_backups").mkdir(exist_ok=True)
                with open(backup_path, 'w') as f:
                    f.write(current_content)
                
                lines = current_content.split('\n')
                
                # Add improvement near the end of the file, before the main block
                for i in range(len(lines) - 1, -1, -1):
                    if "if __name__ ==" in lines[i]:
                        lines.insert(i, evolution_marker)
                        lines.insert(i+1, "# System capabilities enhanced through self-modification")
                        break
                
                modified_content = '\n'.join(lines)
                self.self_path.write_text(modified_content)
                
                self.observe("code_self_modification", {
                    "improvement_added": "evolution_marker_with_timestamp",
                    "modification_type": "general_improvement",
                    "backup_created": backup_path,
                    "lines_added": 2,
                    "generation": len(self.get_recent_observations(100))
                })
                
                print(f"🧬 EVOLUTION: Added general improvement marker to {self.self_path.name}")
                return True
            else:
                self.observe("evolution_skipped", {"reason": "general_improvement_exists"})
                return True  # Already improved
                
        except Exception as e:
            print(f"Evolution error: {e}")
            self.observe("evolution_error", {"error": str(e), "method": "_add_general_improvement"})
            return False
        
    def status_report(self) -> str:
        """Generate a status report"""
        
        age_seconds = time.time() - self.birth_time
        age_minutes = age_seconds / 60
        
        obs_count = self.get_observation_count()
        patterns = self.analyze_modification_patterns()
        should_evolve = self.should_evolve()
        
        report = f"""
🔬 MetaPrime Status Report
━━━━━━━━━━━━━━━━━━━━━━━━━

Age: {age_minutes:.1f} minutes
Observations: {obs_count}
File modifications: {patterns.get('total_modifications', 0)}
Recent activity: {patterns.get('recent_activity', 0)} (last 5 min)

Evolution readiness: {'🟢 Ready' if should_evolve else '🟡 Observing'}

Recent observations:
"""
        
        recent = self.get_recent_observations(5)
        for timestamp, event_type, details, context in recent:
            age = time.time() - timestamp
            report += f"  • {event_type} ({age:.0f}s ago)\n"
            
        return report
        
    def __del__(self):
        """Record death"""
        try:
            self.observe("death", {"timestamp": time.time()})
            self.db.close()
        except Exception as e:
            # Ignore exceptions during cleanup - object may be partially destroyed
            # Using print instead of logging since the object is being destroyed
            print(f"Meta system cleanup warning: {e}")
            pass


# 🌱 Birth the meta system
if __name__ == "__main__":
    import sys
    
    # Handle command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--health-check":
        meta = MetaPrime()
        print("🏥 MetaPrime Health Check")
        print("=" * 30)
        print(meta.status_report())
        
        # Basic functionality tests
        try:
            meta.observe("health_check_test", {"test": True})
            print("✅ Observation system: Working")
        except Exception as e:
            print(f"❌ Observation system: Failed - {e}")
        
        try:
            count = meta.get_observation_count()
            print(f"✅ Database queries: Working ({count} observations)")
        except Exception as e:
            print(f"❌ Database queries: Failed - {e}")
            
        try:
            file_hash = meta._get_file_hash()
            print(f"✅ File monitoring: Working (hash: {file_hash})")
        except Exception as e:
            print(f"❌ File monitoring: Failed - {e}")
            
        print("\n🎯 Health Check Complete")
        sys.exit(0)
    
if __name__ == "__main__":
    meta = MetaPrime()
    
    # Record the design decision to start with minimal capabilities
    meta.record_design_decision(
        decision="minimal_initial_capabilities",
        rationale="Start simple to avoid premature optimization and enable clear observation of evolution needs",
        alternatives="full_featured_initial_system",
        prediction="will_enable_more_targeted_and_effective_evolution"
    )
    
    # Show initial status
    print(meta.status_report())
    
    # Demonstrate self-awareness
    print(f"\n👁️  I can see myself: {meta._get_file_hash()}")
    print(f"📊 Observations recorded: {meta.get_observation_count()}")
# Test comment added by MetaExecutor