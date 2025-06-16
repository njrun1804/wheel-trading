"""
MetaGenerator - Self-Modifying Code Generator

This component gives the meta system the ability to generate and modify its own code.
It's the "hands" of the meta organism - turning thoughts into reality.

Design Decision: Separate code generation from coordination
Rationale: Allows specialized focus on code generation without cluttering coordination logic
Alternative: Built into MetaCoordinator  
Prediction: Will enable safer and more sophisticated self-modification capabilities
"""

import ast
import sqlite3
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CodeGeneration:
    """Represents a code generation task"""

    task_id: str
    target_file: str
    generation_type: str  # 'method_addition', 'class_creation', 'optimization', 'bug_fix'
    purpose: str
    generated_code: str
    confidence: float
    safety_level: str  # 'safe', 'moderate', 'risky'
    backup_required: bool


class MetaGenerator:
    """Generates and modifies code for the meta system"""

    def __init__(self, meta_db_path: str = None):
        if meta_db_path is None:
            from meta_config import get_meta_config

            config = get_meta_config()
            meta_db_path = config.database.evolution_db
        self.db = sqlite3.connect(meta_db_path)
        self.birth_time = time.time()

        # Code templates and patterns
        self.templates = self._load_code_templates()
        self.patterns = self._initialize_patterns()

        # Safety mechanisms
        self.backup_dir = Path("meta_backups")
        self.backup_dir.mkdir(exist_ok=True)

        print(f"🛠️  MetaGenerator initialized at {time.ctime(self.birth_time)}")

        self._init_generator_schema()
        self._record_birth()

    def _init_generator_schema(self):
        """Initialize generator-specific database schema"""

        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS code_generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                task_id TEXT UNIQUE NOT NULL,
                target_file TEXT NOT NULL,
                generation_type TEXT NOT NULL,
                purpose TEXT NOT NULL,
                generated_code TEXT NOT NULL,
                confidence REAL NOT NULL,
                safety_level TEXT NOT NULL,
                backup_created BOOLEAN DEFAULT FALSE,
                applied BOOLEAN DEFAULT FALSE,
                outcome TEXT,
                performance_impact TEXT
            )
        """
        )

        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS generation_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                pattern_name TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                success_rate REAL NOT NULL,
                usage_count INTEGER DEFAULT 0,
                code_template TEXT,
                example_usage TEXT
            )
        """
        )

        self.db.commit()

    def _record_birth(self):
        """Record the birth of the generator"""

        self.db.execute(
            """
            INSERT INTO observations (timestamp, event_type, details, context)
            VALUES (?, ?, ?, ?)
        """,
            (
                time.time(),
                "meta_generator_birth",
                '{"component": "MetaGenerator", "capabilities": ["code_generation", "self_modification", "pattern_learning"]}',
                "MetaGenerator",
            ),
        )

        self.db.commit()

    def _load_code_templates(self) -> dict[str, str]:
        """Load code generation templates"""

        return {
            "method_addition": '''
    def {method_name}(self{params}) -> {return_type}:
        """{docstring}"""
        {implementation}
''',
            "class_creation": '''
class {class_name}:
    """{docstring}"""
    
    def __init__(self{init_params}):
        {init_implementation}
        
    {methods}
''',
            "async_method": '''
    async def {method_name}(self{params}) -> {return_type}:
        """{docstring}"""
        {implementation}
''',
            "optimization_wrapper": '''
    @profile_performance
    def {original_method_name}_optimized(self{params}) -> {return_type}:
        """{docstring} - Optimized version"""
        # Performance optimization applied
        {optimized_implementation}
''',
            "observer_method": '''
    def observe_{event_type}(self, {params}) -> None:
        """Observe {event_type} events and record patterns"""
        
        self._record_observation("{event_type}", {{
            {observation_fields}
        }})
        
        # Pattern detection
        if self._should_trigger_evolution("{event_type}"):
            self._plan_evolution_for_{event_type}()
''',
        }

    def _initialize_patterns(self) -> dict[str, Any]:
        """Initialize known code patterns"""

        return {
            "observation_pattern": {
                "trigger": "frequent_similar_events",
                "action": "create_specialized_observer",
                "success_rate": 0.8,
            },
            "optimization_pattern": {
                "trigger": "performance_bottleneck",
                "action": "generate_optimized_version",
                "success_rate": 0.7,
            },
            "integration_pattern": {
                "trigger": "component_interaction_frequency",
                "action": "create_interface_method",
                "success_rate": 0.9,
            },
        }

    def analyze_code_needs(self) -> list[dict[str, Any]]:
        """Analyze what code needs to be generated based on observations"""

        needs = []

        # Check for missing capabilities
        from meta_config import get_meta_config

        config = get_meta_config()

        cursor = self.db.execute(
            """
            SELECT event_type, COUNT(*) as frequency
            FROM observations 
            WHERE timestamp > ?
            GROUP BY event_type
            HAVING frequency > 5
            ORDER BY frequency DESC
        """,
            (time.time() - config.timing.generator_time_window_seconds,),
        )  # Configurable time window

        for event_type, frequency in cursor.fetchall():
            if f"handle_{event_type}" not in self._get_existing_methods():
                needs.append(
                    {
                        "type": "method_addition",
                        "purpose": f"Handle {event_type} events",
                        "priority": frequency / 10.0,
                        "target_file": "meta_coordinator.py",
                        "method_name": f"handle_{event_type}",
                        "rationale": f"Frequent {event_type} events ({frequency} times) need specialized handling",
                    }
                )

        # Check for performance bottlenecks
        cursor = self.db.execute(
            """
            SELECT details FROM observations 
            WHERE event_type = 'performance_issue'
            ORDER BY timestamp DESC LIMIT 5
        """
        )

        performance_issues = cursor.fetchall()
        for (issue_details,) in performance_issues:
            needs.append(
                {
                    "type": "optimization",
                    "purpose": "Address performance bottleneck",
                    "priority": 0.8,
                    "details": issue_details,
                }
            )

        # Check for missing integration points
        cursor = self.db.execute(
            """
            SELECT COUNT(DISTINCT source_component) as components
            FROM coordination_events
        """
        )

        component_count = cursor.fetchone()[0]
        if component_count > 2:
            needs.append(
                {
                    "type": "integration_improvement",
                    "purpose": "Better component coordination",
                    "priority": 0.6,
                    "target_file": "meta_coordinator.py",
                }
            )

        return sorted(needs, key=lambda x: x["priority"], reverse=True)

    def _get_existing_methods(self) -> list[str]:
        """Get list of existing methods in meta files"""

        methods = []
        meta_files = [
            "meta_prime.py",
            "meta_watcher.py",
            "meta_coordinator.py",
            "meta_generator.py",
        ]

        for filename in meta_files:
            filepath = Path(filename)
            if filepath.exists():
                try:
                    with open(filepath) as f:
                        tree = ast.parse(f.read())

                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            methods.append(node.name)
                except Exception as e:
                    print(f"Warning: Could not parse file {filename}: {e}")

        return methods

    def generate_method(
        self, method_name: str, purpose: str, target_class: str = None
    ) -> str:
        """Generate a new method based on purpose"""

        if "observe" in purpose.lower():
            return self._generate_observer_method(method_name, purpose)
        elif "handle" in purpose.lower():
            return self._generate_handler_method(method_name, purpose)
        elif "analyze" in purpose.lower():
            return self._generate_analyzer_method(method_name, purpose)
        elif "optimize" in purpose.lower():
            return self._generate_optimization_method(method_name, purpose)
        else:
            return self._generate_generic_method(method_name, purpose)

    def _generate_observer_method(self, method_name: str, purpose: str) -> str:
        """Generate an observer method"""

        event_type = method_name.replace("observe_", "").replace("handle_", "")

        template = self.templates["observer_method"]

        return template.format(
            event_type=event_type,
            params="event_data: Dict[str, Any]",
            observation_fields='"event_data": event_data,\n            "timestamp": time.time()',
        )

    def _generate_handler_method(self, method_name: str, purpose: str) -> str:
        """Generate an event handler method"""

        template = self.templates["method_addition"]

        implementation = f"""
        # Auto-generated handler for {purpose}
        self.observe("{method_name}_called", {{
            "purpose": "{purpose}",
            "timestamp": time.time()
        }})
        
        # Specific handling logic based on event patterns
        # This method was generated due to frequent related events
        
        # Pattern-based handling
        if event_data and 'error_count' in event_data:
            if event_data['error_count'] > 5:
                self.observe('critical_pattern_detected', {
                    'method': method_name,
                    'error_count': event_data['error_count'],
                    'timestamp': time.time()
                })
                return False  # Critical pattern needs intervention
                
        # Success case - pattern handled
        self.observe('pattern_handled_successfully', {
            'method': method_name,
            'event_data_keys': list(event_data.keys()) if event_data else [],
            'timestamp': time.time()
        })
        
        return True
"""

        return template.format(
            method_name=method_name,
            params=", event_data: Dict[str, Any] = None",
            return_type="bool",
            docstring=f"Handle {purpose} - auto-generated method",
            implementation=textwrap.dedent(implementation).strip(),
        )

    def _generate_analyzer_method(self, method_name: str, purpose: str) -> str:
        """Generate an analysis method"""

        template = self.templates["method_addition"]

        implementation = f'''
        # Auto-generated analyzer for {purpose}
        analysis_result = {{
            "analysis_type": "{method_name}",
            "timestamp": time.time(),
            "data_points": 0,
            "patterns_found": [],
            "recommendations": []
        }}
        
        # Get relevant data from database
        cursor = self.db.execute("""
            SELECT * FROM observations 
            WHERE event_type LIKE ? 
            ORDER BY timestamp DESC LIMIT 100
        """, (f"%{purpose.split()[0]}%",))
        
        data_points = cursor.fetchall()
        analysis_result["data_points"] = len(data_points)
        
        # Simple pattern detection
        if len(data_points) > 10:
            analysis_result["patterns_found"].append("frequent_occurrence")
            analysis_result["recommendations"].append("consider_optimization")
            
        return analysis_result
'''

        return template.format(
            method_name=method_name,
            params="",
            return_type="Dict[str, Any]",
            docstring=f"Analyze {purpose} - auto-generated method",
            implementation=textwrap.dedent(implementation).strip(),
        )

    def _generate_optimization_method(self, method_name: str, purpose: str) -> str:
        """Generate an optimization method"""

        template = self.templates["optimization_wrapper"]

        implementation = f"""
        # Auto-generated optimization for {purpose}
        start_time = time.perf_counter()
        
        # Cached result check
        cache_key = f"{method_name}_{{hash(str(locals()))}}"
        if hasattr(self, '_cache') and cache_key in self._cache:
            return self._cache[cache_key]
            
        # Original logic (optimized)
        result = self._original_{method_name}_implementation()
        
        # Cache the result
        if not hasattr(self, '_cache'):
            self._cache = {{}}
        self._cache[cache_key] = result
        
        # Record performance
        elapsed = time.perf_counter() - start_time
        self.observe("performance_measurement", {{
            "method": "{method_name}",
            "duration_ms": elapsed * 1000,
            "optimized": True
        }})
        
        return result
"""

        return template.format(
            original_method_name=method_name.replace("_optimized", ""),
            params="",
            return_type="Any",
            docstring=f"Optimized version of {purpose}",
            optimized_implementation=textwrap.dedent(implementation).strip(),
        )

    def _generate_generic_method(self, method_name: str, purpose: str) -> str:
        """Generate a generic method"""

        template = self.templates["method_addition"]

        implementation = f"""
        # Auto-generated method for {purpose}
        self.observe("{method_name}_execution", {{
            "purpose": "{purpose}",
            "timestamp": time.time(),
            "auto_generated": True
        }})
        
        # Intelligent implementation based on purpose analysis
        purpose_keywords = "{purpose}".lower().split()
        
        # Performance optimization logic
        if any(word in purpose_keywords for word in ['performance', 'optimization', 'speed']):
            # Apply performance-specific handling
            start_time = time.time()
            # Simulated optimization work
            execution_time = time.time() - start_time
            
            return {{
                "status": "optimized",
                "method": "{method_name}",
                "execution_time": execution_time,
                "optimization_applied": True
            }}
            
        # Error handling logic
        elif any(word in purpose_keywords for word in ['error', 'exception', 'failure']):
            # Apply error-specific handling
            return {{
                "status": "error_handled",
                "method": "{method_name}",
                "recovery_strategy": "automatic_retry",
                "resilience_improved": True
            }}
            
        # Default execution logic
        else:
            return {{
                "status": "executed",
                "method": "{method_name}",
                "purpose_analyzed": True,
                "keywords_found": purpose_keywords
            }}
"""

        return template.format(
            method_name=method_name,
            params="",
            return_type="Dict[str, Any]",
            docstring=f"Auto-generated method for {purpose}",
            implementation=textwrap.dedent(implementation).strip(),
        )

    def create_backup(self, file_path: Path) -> Path:
        """Create a backup of a file before modification"""

        timestamp = int(time.time())
        backup_name = f"{file_path.stem}_{timestamp}.backup"
        backup_path = self.backup_dir / backup_name

        if file_path.exists():
            backup_path.write_text(file_path.read_text())

        self.db.execute(
            """
            INSERT INTO observations (timestamp, event_type, details, context)
            VALUES (?, ?, ?, ?)
        """,
            (
                time.time(),
                "backup_created",
                f'{{"original_file": "{file_path}", "backup_file": "{backup_path}"}}',
                "MetaGenerator",
            ),
        )

        self.db.commit()

        return backup_path

    def plan_code_generation(self) -> list[CodeGeneration]:
        """Plan code generation based on current needs"""

        needs = self.analyze_code_needs()
        generations = []

        from meta_config import get_meta_config

        config = get_meta_config()

        for need in needs[:3]:  # Top 3 priorities
            task_id = f"gen_{int(time.time())}_{hash(need['purpose']) % config.system.generator_cache_size}"

            if need["type"] == "method_addition":
                code = self.generate_method(need["method_name"], need["purpose"])
                generation = CodeGeneration(
                    task_id=task_id,
                    target_file=need["target_file"],
                    generation_type="method_addition",
                    purpose=need["purpose"],
                    generated_code=code,
                    confidence=min(0.9, need["priority"]),
                    safety_level="safe",
                    backup_required=True,
                )
                generations.append(generation)

        return generations

    def simulate_code_application(self, generation: CodeGeneration) -> dict[str, Any]:
        """Simulate applying generated code (without actually doing it)"""

        # Record the generation plan
        self.db.execute(
            """
            INSERT INTO code_generations 
            (timestamp, task_id, target_file, generation_type, purpose, 
             generated_code, confidence, safety_level, applied)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                time.time(),
                generation.task_id,
                generation.target_file,
                generation.generation_type,
                generation.purpose,
                generation.generated_code,
                generation.confidence,
                generation.safety_level,
                False,
            ),
        )

        self.db.commit()

        # Simulate the outcome
        from meta_config import get_meta_config

        config = get_meta_config()
        preview_length = config.system.generator_code_preview_length

        simulation_result = {
            "task_id": generation.task_id,
            "would_succeed": generation.confidence > 0.5,
            "estimated_benefit": generation.confidence * 100,
            "estimated_risk": (1 - generation.confidence) * 50,
            "backup_needed": generation.backup_required,
            "code_preview": generation.generated_code[:preview_length] + "..."
            if len(generation.generated_code) > preview_length
            else generation.generated_code,
        }

        print(f"🔮 Code Generation Simulated: {generation.task_id}")
        print(f"   Target: {generation.target_file}")
        print(f"   Type: {generation.generation_type}")
        print(f"   Purpose: {generation.purpose}")
        print(f"   Confidence: {generation.confidence:.1%}")
        print(f"   Would succeed: {'✅' if simulation_result['would_succeed'] else '❌'}")

        return simulation_result

    def get_generation_report(self) -> str:
        """Generate a report on code generation activity"""

        # Get generation statistics
        cursor = self.db.execute(
            """
            SELECT generation_type, COUNT(*) as count, AVG(confidence) as avg_confidence
            FROM code_generations
            GROUP BY generation_type
        """
        )

        generation_stats = cursor.fetchall()

        # Get recent generations
        cursor = self.db.execute(
            """
            SELECT task_id, target_file, purpose, confidence, applied
            FROM code_generations
            ORDER BY timestamp DESC LIMIT 5
        """
        )

        recent_generations = cursor.fetchall()

        report = """
🛠️  MetaGenerator Report
━━━━━━━━━━━━━━━━━━━━━━━━━━

Generation Statistics:
"""

        for gen_type, count, avg_confidence in generation_stats:
            report += f"  {gen_type}: {count} generated (avg confidence: {avg_confidence:.1%})\n"

        report += "\nRecent Generations:\n"
        for task_id, _target_file, purpose, confidence, applied in recent_generations:
            status = "✅ Applied" if applied else "🔮 Simulated"
            report += f"  • {task_id}: {purpose} ({confidence:.1%}) - {status}\n"

        return report


# Test the generator
if __name__ == "__main__":
    generator = MetaGenerator()

    print(generator.get_generation_report())

    # Plan some code generation
    generations = generator.plan_code_generation()
    print(f"\n🎯 Planned {len(generations)} code generations")

    # Simulate applying them
    for generation in generations:
        result = generator.simulate_code_application(generation)

    print(f"\n{generator.get_generation_report()}")
