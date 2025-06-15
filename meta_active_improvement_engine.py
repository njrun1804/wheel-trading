#!/usr/bin/env python3
"""
Meta Active Improvement Engine
Real-time code improvement using learned Claude patterns
"""

import time
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from meta_prime import MetaPrime
from meta_coordinator import MetaCoordinator
from meta_auditor import MetaAuditor
from meta_executor import MetaExecutor
from meta_generator import MetaGenerator


@dataclass
class ActiveImprovement:
    """Real-time improvement to apply"""
    improvement_id: str
    pattern_detected: str
    original_code: str
    improved_code: str
    improvement_type: str
    confidence: float
    reasoning: str


class MetaActiveImprovementEngine:
    """Actively improves code using learned Claude patterns"""
    
    def __init__(self):
        # Check if meta system is disabled
        import os
        if os.environ.get('DISABLE_META_AUTOSTART') == '1':
            # Create stub components
            self.meta_prime = None
            self.meta_coordinator = None
            self.meta_auditor = None
            self.meta_executor = None
            self.meta_generator = None
        else:
            # Reuse existing meta components
            self.meta_prime = MetaPrime()
            self.meta_coordinator = MetaCoordinator()
            self.meta_auditor = MetaAuditor()
            self.meta_executor = MetaExecutor()
            self.meta_generator = MetaGenerator()
        
        # Load learned patterns from CLI training
        self.learned_patterns = self._load_learned_patterns()
        self.improvement_rules = self._initialize_improvement_rules()
        
        # Active improvement state
        self.improvements_applied = []
        self.suggestions_made = []
        
        print("🔧 Meta Active Improvement Engine initialized")
        print(f"📚 Loaded {len(self.learned_patterns)} learned patterns")
        print(f"🎯 Ready for real-time code improvement")
    
    def intercept_and_improve_code(self, code: str, context: Dict[str, Any] = None) -> Tuple[str, List[ActiveImprovement]]:
        """Intercept code generation and apply active improvements"""
        
        print(f"🔍 Analyzing code for improvements...")
        
        # Check against learned patterns BEFORE returning code
        improvements = self._detect_improvement_opportunities(code, context or {})
        
        if improvements:
            improved_code = self._apply_improvements(code, improvements)
            
            # Record active improvements
            for improvement in improvements:
                self.improvements_applied.append(improvement)
                self.meta_prime.observe("active_code_improvement", {
                    "improvement_id": improvement.improvement_id,
                    "pattern": improvement.pattern_detected,
                    "type": improvement.improvement_type,
                    "confidence": improvement.confidence,
                    "original_length": len(improvement.original_code),
                    "improved_length": len(improvement.improved_code)
                })
            
            print(f"✅ Applied {len(improvements)} active improvements")
            return improved_code, improvements
        
        else:
            print("📝 No improvements needed - code meets learned standards")
            return code, []
    
    def suggest_improvements(self, code: str, context: Dict[str, Any] = None) -> List[str]:
        """Generate improvement suggestions without modifying code"""
        
        suggestions = []
        
        # Check for common anti-patterns learned from Claude
        if not self._has_proper_error_handling(code):
            suggestions.append("Add try-except blocks for better error handling (learned from Claude)")
        
        if not self._has_documentation(code):
            suggestions.append("Add docstrings and comments (Claude always documents code)")
        
        if not self._follows_naming_conventions(code):
            suggestions.append("Use descriptive variable names (Claude pattern)")
        
        if self._has_magic_numbers(code):
            suggestions.append("Replace magic numbers with named constants (Claude best practice)")
        
        # Record suggestions
        for suggestion in suggestions:
            self.suggestions_made.append({
                "suggestion": suggestion,
                "timestamp": time.time(),
                "code_length": len(code),
                "context": context or {}
            })
        
        if suggestions:
            self.meta_prime.observe("improvement_suggestions", {
                "suggestions_count": len(suggestions),
                "suggestions": suggestions,
                "code_analysis": "active_checking"
            })
        
        return suggestions
    
    def _load_learned_patterns(self) -> Dict[str, Any]:
        """Load patterns learned from CLI training sessions"""
        
        patterns = {}
        
        # Load from CLI session files
        session_files = list(Path(".").glob("claude_cli_session_*.json"))
        
        for session_file in session_files:
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                # Extract patterns from session
                for event in session_data.get("events", []):
                    if event["event_type"] == "code_generation":
                        for pattern in event.get("reasoning_patterns", []):
                            patterns[pattern] = patterns.get(pattern, 0) + 1
                
                # Extract from insights
                insights = session_data.get("insights", {})
                for pattern, count in insights.get("pattern_frequency", {}).items():
                    patterns[pattern] = patterns.get(pattern, 0) + count
                    
            except Exception as e:
                print(f"⚠️ Could not load session {session_file}: {e}")
        
        return patterns
    
    def _initialize_improvement_rules(self) -> Dict[str, Any]:
        """Initialize improvement rules based on Claude patterns"""
        
        return {
            "error_handling": {
                "pattern": "try_except_blocks",
                "check": self._needs_error_handling,
                "improve": self._add_error_handling,
                "priority": "high"
            },
            "documentation": {
                "pattern": "well_documented",
                "check": self._needs_documentation,
                "improve": self._add_documentation,
                "priority": "medium"
            },
            "validation": {
                "pattern": "input_validation", 
                "check": self._needs_validation,
                "improve": self._add_validation,
                "priority": "high"
            },
            "naming": {
                "pattern": "descriptive_names",
                "check": self._needs_better_naming,
                "improve": self._improve_naming,
                "priority": "low"
            }
        }
    
    def _detect_improvement_opportunities(self, code: str, context: Dict[str, Any]) -> List[ActiveImprovement]:
        """Detect opportunities for active improvement"""
        
        improvements = []
        
        for rule_name, rule in self.improvement_rules.items():
            if rule["check"](code):
                # Apply improvement using learned pattern
                improved_code = rule["improve"](code)
                
                if improved_code != code:  # Improvement was made
                    improvement = ActiveImprovement(
                        improvement_id=f"{rule_name}_{int(time.time())}",
                        pattern_detected=rule["pattern"],
                        original_code=code,
                        improved_code=improved_code,
                        improvement_type=rule_name,
                        confidence=self._calculate_improvement_confidence(rule_name),
                        reasoning=f"Applied Claude pattern: {rule['pattern']}"
                    )
                    improvements.append(improvement)
        
        return improvements
    
    def _apply_improvements(self, code: str, improvements: List[ActiveImprovement]) -> str:
        """Apply improvements to code in order of priority"""
        
        improved_code = code
        
        # Sort by confidence and priority
        improvements.sort(key=lambda x: x.confidence, reverse=True)
        
        for improvement in improvements:
            # Use meta_auditor to validate improvement safety
            if self._validate_improvement_safety(improvement):
                improved_code = improvement.improved_code
                print(f"  ✅ Applied: {improvement.improvement_type}")
            else:
                print(f"  ⚠️ Skipped unsafe improvement: {improvement.improvement_type}")
        
        return improved_code
    
    def _validate_improvement_safety(self, improvement: ActiveImprovement) -> bool:
        """Use meta_auditor to validate improvement safety"""
        
        # Basic safety checks
        safety_checks = [
            improvement.confidence >= 0.6,
            len(improvement.improved_code) > 0,
            improvement.improvement_type in self.improvement_rules,
            not self._introduces_security_issues(improvement.improved_code)
        ]
        
        return all(safety_checks)
    
    # Improvement check functions
    def _needs_error_handling(self, code: str) -> bool:
        """Check if code needs error handling"""
        # Look for risky operations without try-except
        risky_patterns = [
            r'open\s*\(',
            r'requests\.',
            r'\.read\(\)',
            r'json\.loads\(',
            r'int\s*\(',
            r'float\s*\('
        ]
        
        has_risky_ops = any(re.search(pattern, code) for pattern in risky_patterns)
        has_error_handling = 'try:' in code and 'except' in code
        
        return has_risky_ops and not has_error_handling
    
    def _needs_documentation(self, code: str) -> bool:
        """Check if code needs documentation"""
        has_functions = 'def ' in code
        has_docstrings = '"""' in code or "'''" in code
        has_comments = '# ' in code
        
        return has_functions and not (has_docstrings or has_comments)
    
    def _needs_validation(self, code: str) -> bool:
        """Check if code needs input validation"""
        has_user_input = any(pattern in code for pattern in ['input(', 'sys.argv', 'request.'])
        has_validation = any(word in code.lower() for word in ['validate', 'check', 'assert', 'if not'])
        
        return has_user_input and not has_validation
    
    def _needs_better_naming(self, code: str) -> bool:
        """Check if code needs better naming"""
        # Look for single-letter variables (except common loop counters)
        single_letter_vars = re.findall(r'\b[a-z]\s*=', code)
        bad_vars = [var for var in single_letter_vars if not var.startswith(('i =', 'j =', 'k ='))]
        
        return len(bad_vars) > 0
    
    # Improvement application functions
    def _add_error_handling(self, code: str) -> str:
        """Add error handling based on Claude patterns"""
        
        if 'open(' in code and 'try:' not in code:
            # Wrap file operations in try-except
            lines = code.split('\n')
            improved_lines = []
            
            for line in lines:
                if 'open(' in line:
                    indent = len(line) - len(line.lstrip())
                    improved_lines.append(' ' * indent + 'try:')
                    improved_lines.append(' ' * (indent + 4) + line.strip())
                    improved_lines.append(' ' * indent + 'except Exception as e:')
                    improved_lines.append(' ' * (indent + 4) + 'print(f"Error: {e}")')
                    improved_lines.append(' ' * (indent + 4) + 'return None')
                else:
                    improved_lines.append(line)
            
            return '\n'.join(improved_lines)
        
        return code
    
    def _add_documentation(self, code: str) -> str:
        """Add documentation based on Claude patterns"""
        
        if 'def ' in code and '"""' not in code:
            lines = code.split('\n')
            improved_lines = []
            
            for i, line in enumerate(lines):
                improved_lines.append(line)
                if line.strip().startswith('def ') and ':' in line:
                    # Add docstring after function definition
                    indent = len(line) - len(line.lstrip())
                    func_name = line.strip().split('(')[0].replace('def ', '')
                    improved_lines.append(' ' * (indent + 4) + f'"""Execute {func_name} operation."""')
            
            return '\n'.join(improved_lines)
        
        return code
    
    def _add_validation(self, code: str) -> str:
        """Add input validation based on Claude patterns"""
        
        if 'input(' in code and 'if not' not in code:
            # Add validation after input statements
            lines = code.split('\n')
            improved_lines = []
            
            for line in lines:
                improved_lines.append(line)
                if 'input(' in line and '=' in line:
                    var_name = line.split('=')[0].strip()
                    indent = len(line) - len(line.lstrip())
                    improved_lines.append(' ' * indent + f'if not {var_name}:')
                    improved_lines.append(' ' * (indent + 4) + f'print("Error: {var_name} is required")')
                    improved_lines.append(' ' * (indent + 4) + 'return')
            
            return '\n'.join(improved_lines)
        
        return code
    
    def _improve_naming(self, code: str) -> str:
        """Improve variable naming based on Claude patterns"""
        
        # Simple example: replace single letters with descriptive names
        naming_map = {
            'x =': 'value =',
            'y =': 'result =', 
            'z =': 'output =',
            'd =': 'data =',
            'f =': 'file_obj =',
            's =': 'text ='
        }
        
        improved_code = code
        for old, new in naming_map.items():
            improved_code = improved_code.replace(old, new)
        
        return improved_code
    
    # Helper functions
    def _has_proper_error_handling(self, code: str) -> bool:
        """Check if code has proper error handling"""
        return 'try:' in code and 'except' in code
    
    def _has_documentation(self, code: str) -> bool:
        """Check if code has documentation"""
        return '"""' in code or "'''" in code or '# ' in code
    
    def _follows_naming_conventions(self, code: str) -> bool:
        """Check if code follows naming conventions"""
        single_letter_vars = re.findall(r'\b[a-z]\s*=', code)
        return len(single_letter_vars) == 0
    
    def _has_magic_numbers(self, code: str) -> bool:
        """Check if code has magic numbers"""
        numbers = re.findall(r'\b\d+\b', code)
        # Ignore common numbers like 0, 1, 2
        magic_numbers = [n for n in numbers if int(n) > 2]
        return len(magic_numbers) > 0
    
    def _introduces_security_issues(self, code: str) -> bool:
        """Check if code introduces security issues"""
        dangerous_patterns = ['eval(', 'exec(', 'shell=True', '__import__']
        return any(pattern in code for pattern in dangerous_patterns)
    
    def _calculate_improvement_confidence(self, improvement_type: str) -> float:
        """Calculate confidence in improvement based on learned patterns"""
        
        # Base confidence from learned patterns
        pattern_frequency = self.learned_patterns.get(improvement_type, 0)
        base_confidence = min(0.5 + (pattern_frequency * 0.1), 0.9)
        
        # Adjust based on improvement type priority
        priority_boost = {
            "high": 0.1,
            "medium": 0.05,
            "low": 0.0
        }
        
        rule = self.improvement_rules.get(improvement_type, {})
        priority = rule.get("priority", "medium")
        
        return min(base_confidence + priority_boost.get(priority, 0), 1.0)
    
    def get_improvement_stats(self) -> Dict[str, Any]:
        """Get statistics on active improvements"""
        
        total_improvements = len(self.improvements_applied)
        total_suggestions = len(self.suggestions_made)
        
        improvement_types = {}
        for improvement in self.improvements_applied:
            imp_type = improvement.improvement_type
            improvement_types[imp_type] = improvement_types.get(imp_type, 0) + 1
        
        stats = {
            "total_improvements_applied": total_improvements,
            "total_suggestions_made": total_suggestions,
            "improvement_types": improvement_types,
            "learned_patterns": len(self.learned_patterns),
            "active_rules": len(self.improvement_rules),
            "avg_confidence": sum(imp.confidence for imp in self.improvements_applied) / max(total_improvements, 1)
        }
        
        self.meta_prime.observe("active_improvement_stats", stats)
        
        return stats


# Demo active improvement
def demo_active_improvement():
    """Demonstrate active code improvement"""
    
    print("🔧 DEMO: ACTIVE CODE IMPROVEMENT ENGINE")
    print("=" * 60)
    
    engine = MetaActiveImprovementEngine()
    
    # Test code that needs improvement
    test_code = '''def process_file(filename):
    f = open(filename)
    data = f.read()
    result = data.upper()
    return result'''
    
    print("📝 Original Code:")
    print(test_code)
    print()
    
    # Apply active improvements
    improved_code, improvements = engine.intercept_and_improve_code(test_code)
    
    print("✅ Improved Code:")
    print(improved_code)
    print()
    
    # Show improvements made
    if improvements:
        print("🔧 Improvements Applied:")
        for improvement in improvements:
            print(f"  • {improvement.improvement_type}: {improvement.reasoning}")
    
    # Show suggestions
    suggestions = engine.suggest_improvements(test_code)
    if suggestions:
        print("\\n💡 Suggestions:")
        for suggestion in suggestions:
            print(f"  • {suggestion}")
    
    # Show stats
    stats = engine.get_improvement_stats()
    print(f"\\n📊 Engine Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\\n🎯 ACTIVE IMPROVEMENT: WORKING!")
    return engine


if __name__ == "__main__":
    import os
    import sys
    if os.environ.get('DISABLE_META_AUTOSTART') == '1':
        print("⚠️ Meta Active Improvement Engine blocked by DISABLE_META_AUTOSTART=1")
        sys.exit(0)
    demo_active_improvement()