#!/usr/bin/env python3
"""Utility Migration Map - Consolidation Strategy for Agent 6"""

import json


class UtilityMigrationPlanner:
    def __init__(self):
        # Load analysis results
        with open("utility_analysis_results.json") as f:
            self.analysis = json.load(f)

        self.migration_map = {}
        self.quick_wins = []

    def create_migration_map(self):
        """Create comprehensive migration strategy"""

        # Target directory structure
        target_structure = {
            "src/unity_wheel/memory/": {
                "pool.py": "Connection pooling and memory pools",
                "cache.py": "All caching utilities",
                "optimizer.py": "Memory optimization and cleanup",
                "monitor.py": "Memory monitoring and pressure detection",
            },
            "src/unity_wheel/utils/": {
                "logging.py": "Unified logging with performance optimizations",
                "validation.py": "All validation utilities with caching",
                "formatting.py": "Format, encode, decode utilities",
                "conversion.py": "Type conversion and normalization",
            },
            "src/unity_wheel/core/": {
                "error_handling.py": "Unified error handling and recovery",
                "config.py": "Configuration management utilities",
            },
            "src/unity_wheel/io/": {
                "file_operations.py": "Load, save, read, write utilities",
                "serialization.py": "JSON, pickle, parquet utilities",
            },
        }

        # Map existing utilities to new locations
        self.migration_map = {
            "memory_utilities": self._map_memory_utilities(),
            "logging_utilities": self._map_logging_utilities(),
            "validation_utilities": self._map_validation_utilities(),
            "io_utilities": self._map_io_utilities(),
        }

        # Identify quick wins
        self._identify_quick_wins()

    def _map_memory_utilities(self) -> dict:
        """Map memory utilities to consolidated structure"""
        memory_utils = self.analysis["utilities_by_category"].get("memory", [])

        mapping = {"pool.py": [], "cache.py": [], "optimizer.py": [], "monitor.py": []}

        for util in memory_utils:
            name = util["name"].lower()
            if any(kw in name for kw in ["pool", "connection"]):
                mapping["pool.py"].append(util)
            elif any(kw in name for kw in ["cache", "buffer"]):
                mapping["cache.py"].append(util)
            elif any(kw in name for kw in ["optimize", "cleanup", "free"]):
                mapping["optimizer.py"].append(util)
            elif any(kw in name for kw in ["monitor", "pressure", "usage"]):
                mapping["monitor.py"].append(util)

        return mapping

    def _map_logging_utilities(self) -> dict:
        """Map logging utilities"""
        logging_utils = self.analysis["utilities_by_category"].get("logging", [])

        return {"unified_logging.py": logging_utils}

    def _map_validation_utilities(self) -> dict:
        """Map validation utilities"""
        validation_utils = self.analysis["utilities_by_category"].get("validation", [])

        mapping = {
            "core_validators.py": [],
            "data_validators.py": [],
            "schema_validators.py": [],
        }

        for util in validation_utils:
            name = util["name"].lower()
            if any(kw in name for kw in ["data", "input", "output"]):
                mapping["data_validators.py"].append(util)
            elif any(kw in name for kw in ["schema", "structure", "format"]):
                mapping["schema_validators.py"].append(util)
            else:
                mapping["core_validators.py"].append(util)

        return mapping

    def _map_io_utilities(self) -> dict:
        """Map I/O utilities"""
        io_utils = self.analysis["utilities_by_category"].get("io", [])

        mapping = {"file_operations.py": [], "serialization.py": []}

        for util in io_utils:
            name = util["name"].lower()
            if any(kw in name for kw in ["json", "pickle", "parquet", "serialize"]):
                mapping["serialization.py"].append(util)
            else:
                mapping["file_operations.py"].append(util)

        return mapping

    def _identify_quick_wins(self):
        """Identify immediate consolidation opportunities"""
        duplicates = self.analysis["duplicates"]

        # Priority 1: Exact duplicates
        for func_name, files in duplicates.items():
            if len(files) > 2:
                self.quick_wins.append(
                    {
                        "type": "duplicate_elimination",
                        "function": func_name,
                        "files": files,
                        "priority": "HIGH",
                        "effort": "LOW",
                        "impact": f"Remove {len(files)-1} duplicate implementations",
                    }
                )

        # Priority 2: Similar function consolidation
        memory_functions = [
            u["name"] for u in self.analysis["utilities_by_category"].get("memory", [])
        ]

        # Find similar memory management functions
        cleanup_functions = [f for f in memory_functions if "cleanup" in f.lower()]
        if len(cleanup_functions) > 5:
            self.quick_wins.append(
                {
                    "type": "function_consolidation",
                    "category": "memory_cleanup",
                    "functions": cleanup_functions[:10],  # Top 10
                    "priority": "HIGH",
                    "effort": "MEDIUM",
                    "impact": f"Consolidate {len(cleanup_functions)} cleanup functions into unified API",
                }
            )

        # Priority 3: Import simplification
        high_import_files = self.analysis.get("high_import_files", [])
        for file, import_count in high_import_files[:5]:
            self.quick_wins.append(
                {
                    "type": "import_reduction",
                    "file": file,
                    "current_imports": import_count,
                    "priority": "MEDIUM",
                    "effort": "MEDIUM",
                    "impact": f"Reduce {import_count} imports to ~10-15",
                }
            )

    def generate_migration_plan(self):
        """Generate actionable migration plan"""
        print("UTILITY CONSOLIDATION MIGRATION PLAN")
        print("=" * 50)

        print("\n## PHASE 1: QUICK WINS (1-2 days)")
        for i, win in enumerate(self.quick_wins[:5], 1):
            print(f"\n{i}. {win['type'].upper()}")
            print(f"   Priority: {win['priority']}")
            print(f"   Effort: {win['effort']}")
            print(f"   Impact: {win['impact']}")
            if win["type"] == "duplicate_elimination":
                print(f"   Function: {win['function']}")
                print(f"   Files affected: {len(win['files'])}")

        print("\n## PHASE 2: MEMORY CONSOLIDATION (2-3 days)")
        print("\nTarget structure: src/unity_wheel/memory/")
        memory_map = self.migration_map["memory_utilities"]
        for target_file, utils in memory_map.items():
            if utils:
                print(f"\n  {target_file}: {len(utils)} functions")
                # Show top 3 functions
                for util in utils[:3]:
                    print(f"    - {util['name']} (from {util['file']})")
                if len(utils) > 3:
                    print(f"    ... and {len(utils)-3} more")

        print("\n## PHASE 3: LOGGING CONSOLIDATION (1 day)")
        logging_utils = self.migration_map["logging_utilities"]["unified_logging.py"]
        print(f"\nConsolidate {len(logging_utils)} logging functions into unified API")
        print("Key improvements:")
        print("  - Single logger factory with caching")
        print("  - Lazy formatting for performance")
        print("  - Structured logging with context")
        print("  - Async logging for high throughput")

        print("\n## PHASE 4: VALIDATION CONSOLIDATION (2 days)")
        validation_map = self.migration_map["validation_utilities"]
        for target_file, utils in validation_map.items():
            if utils:
                print(f"\n  {target_file}: {len(utils)} functions")

        # Generate scripts
        self._generate_migration_scripts()

    def _generate_migration_scripts(self):
        """Generate actual migration scripts"""

        # Script 1: Remove duplicates
        duplicate_script = """#!/usr/bin/env python3
# Auto-generated duplicate removal script

import os
from pathlib import Path

# Duplicates to remove (keeping first occurrence)
duplicates_to_remove = [
"""

        for func_name, files in list(self.analysis["duplicates"].items())[:10]:
            if len(files) > 1:
                # Keep first, remove rest
                for file in files[1:]:
                    duplicate_script += f'    "{file}",  # {func_name}\n'

        duplicate_script += """
]

# TODO: Implement safe removal with backup
"""

        with open("remove_duplicates.py", "w") as f:
            f.write(duplicate_script)

        print("\n\nGenerated migration scripts:")
        print("  - remove_duplicates.py")
        print("  - consolidate_memory.py (TODO)")
        print("  - unify_logging.py (TODO)")


if __name__ == "__main__":
    planner = UtilityMigrationPlanner()
    planner.create_migration_map()
    planner.generate_migration_plan()
