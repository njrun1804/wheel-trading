#!/usr/bin/env python3
"""
Meta Fast Pattern Cache
Ultra-fast pattern lookup for real-time code interception
"""

import hashlib
import json
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from meta_prime import MetaPrime


@dataclass
class FastPattern:
    """Cached pattern for fast lookup"""

    pattern_id: str
    pattern_type: str
    confidence: float
    frequency: int
    last_seen: float
    improvement_rule: str
    code_signature: str


@dataclass
class ImprovementRule:
    """Fast improvement rule"""

    rule_id: str
    pattern_trigger: str
    code_check: str  # Regex or simple string check
    improvement_action: str
    confidence: float
    execution_time_ms: float


class MetaFastPatternCache:
    """Ultra-fast pattern cache for real-time code improvement"""

    def __init__(self, cache_size: int = 10000):
        # Only create MetaPrime if not disabled
        import os

        if os.environ.get("DISABLE_META_AUTOSTART") == "1":
            self.meta_prime = None
        else:
            self.meta_prime = MetaPrime()

        # High-speed caches
        self.pattern_cache: dict[str, FastPattern] = {}
        self.improvement_rules: dict[str, ImprovementRule] = {}
        self.code_signature_cache: dict[str, list[str]] = {}

        # Hot pattern tracking (most frequently used)
        self.hot_patterns: deque = deque(maxlen=100)
        self.pattern_frequency: dict[str, int] = defaultdict(int)

        # Performance optimization
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        self.avg_lookup_time = 0.0

        # Thread safety
        self.cache_lock = threading.RLock()

        print("⚡ Fast Pattern Cache initializing...")
        self._precompute_patterns()
        self._build_improvement_rules()
        print(
            f"✅ Cache ready: {len(self.pattern_cache)} patterns, {len(self.improvement_rules)} rules"
        )

    def _precompute_patterns(self):
        """Precompute patterns from meta DB for fast lookup"""

        # Skip if meta system is disabled
        if self.meta_prime is None:
            print("⚠️ Pattern precomputation skipped (meta system disabled)")
            return

        start_time = time.perf_counter()

        try:
            conn = sqlite3.connect("meta_evolution.db")
            cursor = conn.cursor()

            # Get pattern data efficiently with a single query
            cursor.execute(
                """
                SELECT event_type, details, timestamp 
                FROM observations 
                WHERE event_type LIKE '%pattern%' OR event_type LIKE '%code%' OR event_type LIKE '%claude%'
                ORDER BY timestamp DESC 
                LIMIT 5000
            """
            )

            pattern_stats = defaultdict(
                lambda: {"count": 0, "confidence_sum": 0, "last_seen": 0}
            )

            for _event_type, details_json, timestamp in cursor.fetchall():
                try:
                    details = json.loads(details_json) if details_json else {}

                    # Extract pattern info
                    if "pattern_type" in details:
                        pattern_type = details["pattern_type"]
                        confidence = details.get("confidence", 0.5)

                        pattern_stats[pattern_type]["count"] += 1
                        pattern_stats[pattern_type]["confidence_sum"] += confidence
                        pattern_stats[pattern_type]["last_seen"] = max(
                            pattern_stats[pattern_type]["last_seen"], timestamp
                        )

                    # Extract reasoning patterns
                    if "reasoning_patterns" in details:
                        for pattern in details["reasoning_patterns"]:
                            pattern_stats[pattern]["count"] += 1
                            pattern_stats[pattern]["confidence_sum"] += 0.7  # Default
                            pattern_stats[pattern]["last_seen"] = max(
                                pattern_stats[pattern]["last_seen"], timestamp
                            )

                except (json.JSONDecodeError, KeyError):
                    continue

            conn.close()

            # Build fast patterns
            for pattern_type, stats in pattern_stats.items():
                if stats["count"] >= 2:  # Only patterns seen multiple times
                    avg_confidence = stats["confidence_sum"] / stats["count"]

                    fast_pattern = FastPattern(
                        pattern_id=hashlib.md5(pattern_type.encode()).hexdigest()[:8],
                        pattern_type=pattern_type,
                        confidence=avg_confidence,
                        frequency=stats["count"],
                        last_seen=stats["last_seen"],
                        improvement_rule=self._map_pattern_to_rule(pattern_type),
                        code_signature=self._generate_code_signature(pattern_type),
                    )

                    self.pattern_cache[pattern_type] = fast_pattern

            # Update hot patterns (most frequent)
            sorted_patterns = sorted(
                pattern_stats.items(), key=lambda x: x[1]["count"], reverse=True
            )
            for pattern_type, _ in sorted_patterns[:50]:
                if pattern_type in self.pattern_cache:
                    self.hot_patterns.append(pattern_type)

            elapsed = (time.perf_counter() - start_time) * 1000
            print(f"⚡ Pattern precomputation: {elapsed:.1f}ms")

        except Exception as e:
            print(f"⚠️ Error precomputing patterns: {e}")

    def _build_improvement_rules(self):
        """Build fast improvement rules from patterns"""

        rule_mappings = {
            "well_documented": {
                "check": "def " in "{code}" and '"""' not in "{code}",
                "action": "add_docstring",
                "confidence": 0.8,
            },
            "error_handling": {
                "check": "open(" in "{code}" and "try:" not in "{code}",
                "action": "wrap_in_try_except",
                "confidence": 0.9,
            },
            "systematic_reasoning": {
                "check": "if " in "{code}" and "else" not in "{code}",
                "action": "add_else_clause",
                "confidence": 0.6,
            },
            "safety_conscious": {
                "check": "input(" in "{code}" and "validate" not in "{code}",
                "action": "add_input_validation",
                "confidence": 0.8,
            },
            "solution_oriented": {
                "check": len("{code}") < 50,
                "action": "add_comprehensive_solution",
                "confidence": 0.5,
            },
        }

        for pattern_type, rule_config in rule_mappings.items():
            if pattern_type in self.pattern_cache:
                pattern = self.pattern_cache[pattern_type]

                rule = ImprovementRule(
                    rule_id=f"rule_{pattern.pattern_id}",
                    pattern_trigger=pattern_type,
                    code_check=rule_config["check"],
                    improvement_action=rule_config["action"],
                    confidence=min(pattern.confidence, rule_config["confidence"]),
                    execution_time_ms=1.0,  # Target < 1ms per rule
                )

                self.improvement_rules[pattern_type] = rule

        print(f"🎯 Built {len(self.improvement_rules)} fast improvement rules")

    def fast_pattern_lookup(self, code: str, max_time_ms: float = 5.0) -> list[str]:
        """Ultra-fast pattern lookup with time budget"""

        start_time = time.perf_counter()
        patterns_found = []

        with self.cache_lock:
            # Check hot patterns first (most likely hits)
            for pattern_type in self.hot_patterns:
                if (time.perf_counter() - start_time) * 1000 > max_time_ms:
                    break

                if self._matches_pattern_fast(code, pattern_type):
                    patterns_found.append(pattern_type)
                    self._update_pattern_frequency(pattern_type)

            # Check remaining patterns if time allows
            if (time.perf_counter() - start_time) * 1000 < max_time_ms * 0.5:
                for pattern_type in self.pattern_cache:
                    if pattern_type not in self.hot_patterns:
                        if self._matches_pattern_fast(code, pattern_type):
                            patterns_found.append(pattern_type)
                            self._update_pattern_frequency(pattern_type)

        # Update performance metrics
        lookup_time = (time.perf_counter() - start_time) * 1000
        self.avg_lookup_time = (self.avg_lookup_time * 0.9) + (lookup_time * 0.1)

        if patterns_found:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        return patterns_found

    def get_improvement_rules_fast(self, patterns: list[str]) -> list[ImprovementRule]:
        """Get improvement rules for patterns in O(1) time"""

        rules = []
        for pattern_type in patterns:
            if pattern_type in self.improvement_rules:
                rules.append(self.improvement_rules[pattern_type])

        # Sort by confidence for best-first application
        return sorted(rules, key=lambda r: r.confidence, reverse=True)

    def _matches_pattern_fast(self, code: str, pattern_type: str) -> bool:
        """Fast pattern matching using precomputed signatures"""

        # Use code signature for ultra-fast matching
        code_signature = self._generate_code_signature_fast(code)

        if pattern_type in self.pattern_cache:
            pattern = self.pattern_cache[pattern_type]

            # Simple signature matching
            if pattern.code_signature in code_signature:
                return True

            # Fast heuristic checks
            if pattern_type == "well_documented":
                return "def " in code and '"""' not in code
            elif pattern_type == "error_handling":
                return (
                    any(risky in code for risky in ["open(", "requests.", "json.loads"])
                    and "try:" not in code
                )
            elif pattern_type == "systematic_reasoning":
                return code.count("if ") > 0 and "step" in code.lower()
            elif pattern_type == "safety_conscious":
                return (
                    any(word in code.lower() for word in ["input(", "user"])
                    and "validate" not in code.lower()
                )

        return False

    def _generate_code_signature_fast(self, code: str) -> str:
        """Generate fast code signature for pattern matching"""

        # Extract key code elements quickly
        signature_elements = []

        if "def " in code:
            signature_elements.append("FUNC")
        if "class " in code:
            signature_elements.append("CLASS")
        if "try:" in code:
            signature_elements.append("TRY")
        if "open(" in code:
            signature_elements.append("FILE")
        if "input(" in code:
            signature_elements.append("INPUT")
        if '"""' in code or "'''" in code:
            signature_elements.append("DOC")
        if any(word in code for word in ["print(", "log"]):
            signature_elements.append("DEBUG")

        return "|".join(signature_elements)

    def _generate_code_signature(self, pattern_type: str) -> str:
        """Generate code signature for pattern"""

        signatures = {
            "well_documented": "FUNC|DOC",
            "error_handling": "TRY|FILE",
            "systematic_reasoning": "FUNC",
            "safety_conscious": "INPUT",
            "solution_oriented": "FUNC",
        }

        return signatures.get(pattern_type, "")

    def _map_pattern_to_rule(self, pattern_type: str) -> str:
        """Map pattern type to improvement rule"""

        rule_mapping = {
            "well_documented": "add_documentation",
            "error_handling": "add_error_handling",
            "systematic_reasoning": "add_systematic_approach",
            "safety_conscious": "add_safety_checks",
            "solution_oriented": "enhance_solution",
        }

        return rule_mapping.get(pattern_type, "no_action")

    def _update_pattern_frequency(self, pattern_type: str):
        """Update pattern frequency for hot pattern optimization"""

        self.pattern_frequency[pattern_type] += 1

        # Reorder hot patterns based on frequency
        if len(self.pattern_frequency) % 100 == 0:  # Periodic reordering
            top_patterns = sorted(
                self.pattern_frequency.items(), key=lambda x: x[1], reverse=True
            )[:50]

            self.hot_patterns.clear()
            for pattern_type, _ in top_patterns:
                self.hot_patterns.append(pattern_type)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics"""

        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / max(total_requests, 1)) * 100

        return {
            "cache_size": len(self.pattern_cache),
            "improvement_rules": len(self.improvement_rules),
            "hot_patterns": len(self.hot_patterns),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": hit_rate,
            "avg_lookup_time_ms": self.avg_lookup_time,
            "total_pattern_frequency": dict(self.pattern_frequency),
        }

    def warm_cache(self, code_samples: list[str]):
        """Warm the cache with code samples"""

        print("🔥 Warming pattern cache...")
        start_time = time.perf_counter()

        for code in code_samples:
            self.fast_pattern_lookup(code, max_time_ms=2.0)

        elapsed = (time.perf_counter() - start_time) * 1000
        print(f"✅ Cache warmed in {elapsed:.1f}ms")


# Fast code interceptor using the cache
class FastCodeInterceptor:
    """Ultra-fast code interceptor using pattern cache"""

    def __init__(self):
        self.pattern_cache = MetaFastPatternCache()

        # Warm cache with common code patterns
        common_patterns = [
            "def read_file(filename):\n    return open(filename).read()",
            'x = input("Enter value: ")\nprint(x)',
            "def process(data):\n    result = data * 2\n    return result",
            "class Example:\n    def __init__(self):\n        pass",
        ]

        self.pattern_cache.warm_cache(common_patterns)
        print("⚡ Fast Code Interceptor ready")

    def intercept_and_improve(
        self, code: str, time_budget_ms: float = 10.0
    ) -> tuple[str, float]:
        """Intercept and improve code with strict time budget"""

        start_time = time.perf_counter()

        # Fast pattern lookup
        patterns = self.pattern_cache.fast_pattern_lookup(
            code, max_time_ms=time_budget_ms * 0.5
        )

        if not patterns:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return code, elapsed_ms

        # Fast rule application
        rules = self.pattern_cache.get_improvement_rules_fast(patterns)
        improved_code = self._apply_rules_fast(code, rules, time_budget_ms * 0.5)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return improved_code, elapsed_ms

    def _apply_rules_fast(
        self, code: str, rules: list[ImprovementRule], time_budget_ms: float
    ) -> str:
        """Apply improvement rules with time budget"""

        start_time = time.perf_counter()
        improved_code = code

        for rule in rules:
            if (time.perf_counter() - start_time) * 1000 > time_budget_ms:
                break

            # Fast rule application
            if (
                rule.improvement_action == "add_docstring"
                and "def " in code
                and '"""' not in code
            ):
                improved_code = self._add_docstring_fast(improved_code)
            elif (
                rule.improvement_action == "wrap_in_try_except"
                and "open(" in code
                and "try:" not in code
            ):
                improved_code = self._add_try_except_fast(improved_code)

        return improved_code

    def _add_docstring_fast(self, code: str) -> str:
        """Add docstring quickly"""
        lines = code.split("\n")
        for i, line in enumerate(lines):
            if "def " in line and ":" in line:
                lines.insert(i + 1, '    """Function docstring."""')
                break
        return "\n".join(lines)

    def _add_try_except_fast(self, code: str) -> str:
        """Add try-except quickly"""
        if "open(" in code and "try:" not in code:
            return (
                f"try:\n    {code}\nexcept Exception as e:\n    print(f'Error: {{e}}')"
            )
        return code


def demo_fast_interception():
    """Demonstrate ultra-fast code interception"""

    print("⚡ DEMO: ULTRA-FAST CODE INTERCEPTION")
    print("=" * 60)

    interceptor = FastCodeInterceptor()

    test_codes = [
        "def read_file(f): return open(f).read()",
        'x = input("Name: "); print(x)',
        "def calc(a, b): return a + b",
        "data = json.loads(response.text); print(data)",
    ]

    total_time = 0
    improvements_made = 0

    for i, code in enumerate(test_codes, 1):
        print(f"\n{i}️⃣ Testing: {code}")

        improved_code, elapsed_ms = interceptor.intercept_and_improve(
            code, time_budget_ms=10.0
        )
        total_time += elapsed_ms

        if improved_code != code:
            improvements_made += 1
            print(f"   ✅ Improved in {elapsed_ms:.2f}ms")
            print(f"   → {improved_code[:60]}...")
        else:
            print(f"   📝 No changes needed ({elapsed_ms:.2f}ms)")

    # Show performance stats
    stats = interceptor.pattern_cache.get_cache_stats()

    print("\n⚡ PERFORMANCE RESULTS:")
    print(f"   Total time: {total_time:.2f}ms")
    print(f"   Avg per intercept: {total_time/len(test_codes):.2f}ms")
    print(f"   Improvements made: {improvements_made}/{len(test_codes)}")
    print(f"   Cache hit rate: {stats['hit_rate_percent']:.1f}%")
    print(f"   Avg lookup time: {stats['avg_lookup_time_ms']:.2f}ms")

    target_time = 10.0
    if total_time / len(test_codes) < target_time:
        print(f"✅ Performance target met: < {target_time}ms per intercept")
    else:
        print(f"⚠️ Performance target missed: > {target_time}ms per intercept")

    return interceptor


if __name__ == "__main__":
    demo_fast_interception()
