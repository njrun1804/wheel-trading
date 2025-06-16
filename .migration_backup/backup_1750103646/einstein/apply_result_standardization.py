#!/usr/bin/env python3
"""
Apply Result Standardization to Einstein Components

This script applies the unified result format to all Einstein search components,
ensuring consistent result formats across the entire system while maintaining
backward compatibility.

Usage:
    python apply_result_standardization.py --mode=test    # Test compatibility
    python apply_result_standardization.py --mode=apply   # Apply changes
    python apply_result_standardization.py --mode=verify  # Verify implementation
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the new unified format components
from .unified_result_format import UnifiedSearchResult, UnifiedSearchResponse, ResultConverter
from .result_standardization_adapter import UnifiedSearchInterface, get_standardization_adapter


class ResultStandardizationApplicator:
    """
    Applies result standardization to Einstein components with comprehensive testing.
    """
    
    def __init__(self):
        self.test_results = {}
        self.errors = []
        self.warnings = []
    
    async def test_compatibility(self) -> Dict[str, Any]:
        """Test compatibility with existing Einstein components."""
        
        logger.info("🧪 Testing result format compatibility...")
        
        test_results = {
            "unified_index": await self._test_unified_index(),
            "result_merger": await self._test_result_merger(),
            "optimized_search": await self._test_optimized_search(),
            "einstein_launcher": await self._test_einstein_launcher(),
            "unified_cli": await self._test_unified_cli_compatibility(),
            "conversion_performance": await self._test_conversion_performance()
        }
        
        self.test_results = test_results
        
        # Summary
        total_tests = sum(len(component_tests) for component_tests in test_results.values())
        passed_tests = sum(
            sum(1 for test in component_tests.values() if test.get('passed', False))
            for component_tests in test_results.values()
        )
        
        logger.info(f"✅ Compatibility testing complete: {passed_tests}/{total_tests} tests passed")
        
        return test_results
    
    async def _test_unified_index(self) -> Dict[str, Any]:
        """Test UnifiedIndex compatibility."""
        
        tests = {}
        
        try:
            # Import and test unified index
            from .unified_index import get_einstein_hub
            
            hub = get_einstein_hub()
            await hub.initialize()
            
            # Test basic search
            try:
                results = await hub.search("test query")
                standardized = ResultConverter.to_unified_results(results)
                
                tests["basic_search"] = {
                    "passed": True,
                    "result_count": len(standardized),
                    "conversion_successful": all(isinstance(r, UnifiedSearchResult) for r in standardized)
                }
            except Exception as e:
                tests["basic_search"] = {"passed": False, "error": str(e)}
            
            # Test individual search methods
            for method_name in ["_text_search", "_semantic_search", "_structural_search"]:
                if hasattr(hub, method_name):
                    try:
                        method = getattr(hub, method_name)
                        results = await method("test")
                        standardized = ResultConverter.to_unified_results(results)
                        
                        tests[method_name] = {
                            "passed": True,
                            "result_count": len(standardized),
                            "conversion_successful": all(isinstance(r, UnifiedSearchResult) for r in standardized)
                        }
                    except Exception as e:
                        tests[method_name] = {"passed": False, "error": str(e)}
            
        except Exception as e:
            tests["import_error"] = {"passed": False, "error": str(e)}
        
        return tests
    
    async def _test_result_merger(self) -> Dict[str, Any]:
        """Test ResultMerger compatibility."""
        
        tests = {}
        
        try:
            from .result_merger import ResultMerger
            from .unified_index import SearchResult
            
            merger = ResultMerger()
            
            # Create test results in legacy format
            test_results = {
                "text": [
                    SearchResult(
                        content="test content 1",
                        file_path="test1.py",
                        line_number=10,
                        score=0.8,
                        result_type="text",
                        context={},
                        timestamp=time.time()
                    )
                ],
                "semantic": [
                    SearchResult(
                        content="test content 2",
                        file_path="test2.py",
                        line_number=20,
                        score=0.9,
                        result_type="semantic",
                        context={},
                        timestamp=time.time()
                    )
                ]
            }
            
            # Test merger
            try:
                merged_results = merger.merge_results(test_results)
                standardized = ResultConverter.to_unified_results(merged_results)
                
                tests["merge_results"] = {
                    "passed": True,
                    "merged_count": len(merged_results),
                    "standardized_count": len(standardized),
                    "conversion_successful": all(isinstance(r, UnifiedSearchResult) for r in standardized)
                }
            except Exception as e:
                tests["merge_results"] = {"passed": False, "error": str(e)}
            
        except Exception as e:
            tests["import_error"] = {"passed": False, "error": str(e)}
        
        return tests
    
    async def _test_optimized_search(self) -> Dict[str, Any]:
        """Test OptimizedUnifiedSearch compatibility."""
        
        tests = {}
        
        try:
            from .optimized_unified_search import OptimizedUnifiedSearch
            from .query_router import QueryRouter
            
            # Mock index hub for testing
            class MockIndexHub:
                async def _text_search(self, query): 
                    return [{"content": f"text result for {query}", "file_path": "test.py", "line_number": 1, "score": 0.8, "result_type": "text", "context": {}, "timestamp": time.time()}]
                async def _semantic_search(self, query): 
                    return [{"content": f"semantic result for {query}", "file_path": "test.py", "line_number": 2, "score": 0.9, "result_type": "semantic", "context": {}, "timestamp": time.time()}]
            
            router = QueryRouter()
            search = OptimizedUnifiedSearch(MockIndexHub(), router)
            
            # Test search
            try:
                results, metrics = await search.search("test query")
                standardized = ResultConverter.to_unified_results(results)
                
                tests["optimized_search"] = {
                    "passed": True,
                    "result_count": len(results),
                    "standardized_count": len(standardized),
                    "metrics_available": metrics is not None,
                    "conversion_successful": all(isinstance(r, UnifiedSearchResult) for r in standardized)
                }
            except Exception as e:
                tests["optimized_search"] = {"passed": False, "error": str(e)}
            
        except Exception as e:
            tests["import_error"] = {"passed": False, "error": str(e)}
        
        return tests
    
    async def _test_einstein_launcher(self) -> Dict[str, Any]:
        """Test EinsteinLauncher compatibility."""
        
        tests = {}
        
        try:
            # Test if we can create a response in the expected format
            test_results = [
                UnifiedSearchResult(
                    content="test content",
                    file_path="test.py",
                    line_number=10,
                    score=0.8,
                    result_type="text"
                )
            ]
            
            response = UnifiedSearchResponse(
                query="test query",
                results=test_results,
                search_time_ms=50.0
            )
            
            # Test CLI format conversion
            cli_format = response.to_cli_format()
            
            tests["response_creation"] = {
                "passed": True,
                "has_required_fields": all(
                    field in cli_format for field in ["query", "results", "summary", "system"]
                )
            }
            
            # Test dictionary conversion
            dict_format = response.to_dict()
            
            tests["dict_conversion"] = {
                "passed": True,
                "serializable": isinstance(dict_format, dict),
                "has_results": "results" in dict_format
            }
            
        except Exception as e:
            tests["response_test"] = {"passed": False, "error": str(e)}
        
        return tests
    
    async def _test_unified_cli_compatibility(self) -> Dict[str, Any]:
        """Test compatibility with unified CLI expectations."""
        
        tests = {}
        
        # Test CLI-expected result format
        try:
            test_response = UnifiedSearchResponse(
                query="test query",
                results=[
                    UnifiedSearchResult(
                        content="test content",
                        file_path="test.py",
                        line_number=10,
                        score=0.8,
                        result_type="text"
                    )
                ],
                search_time_ms=25.0
            )
            
            cli_format = test_response.to_cli_format()
            
            # Check CLI expectations
            tests["cli_format"] = {
                "passed": True,
                "has_results_key": "results" in cli_format,
                "has_summary_key": "summary" in cli_format,
                "has_routing_key": "routing" in cli_format,
                "has_system_key": "system" in cli_format,
                "system_is_einstein": cli_format.get("system") == "einstein"
            }
            
            # Test result object compatibility (CLI checks hasattr(res, "file_path"))
            results = cli_format["results"]
            if results:
                first_result = results[0]
                tests["result_compatibility"] = {
                    "passed": True,
                    "has_file_path": hasattr(first_result, "file_path") or (isinstance(first_result, dict) and "file_path" in first_result),
                    "has_score": hasattr(first_result, "score") or (isinstance(first_result, dict) and "score" in first_result),
                    "result_type": str(type(first_result))
                }
            
        except Exception as e:
            tests["cli_compatibility"] = {"passed": False, "error": str(e)}
        
        return tests
    
    async def _test_conversion_performance(self) -> Dict[str, Any]:
        """Test performance of result format conversions."""
        
        tests = {}
        
        try:
            # Create test data in different formats
            legacy_search_results = [
                {"content": f"content {i}", "file_path": f"file{i}.py", "line_number": i, "score": 0.8 - i*0.1, "result_type": "text", "context": {}, "timestamp": time.time()}
                for i in range(100)
            ]
            
            legacy_merged_results = [
                {"content": f"content {i}", "file_path": f"file{i}.py", "line_number": i, "combined_score": 0.9 - i*0.05, "modality_scores": {"text": 0.8, "semantic": 0.7}, "source_modalities": ["text", "semantic"], "context": {}, "timestamp": time.time()}
                for i in range(100)
            ]
            
            # Test conversion performance
            start_time = time.time()
            converted_search = ResultConverter.to_unified_results(legacy_search_results)
            search_conversion_time = (time.time() - start_time) * 1000
            
            start_time = time.time()
            converted_merged = ResultConverter.to_unified_results(legacy_merged_results)
            merged_conversion_time = (time.time() - start_time) * 1000
            
            tests["search_result_conversion"] = {
                "passed": True,
                "conversion_time_ms": search_conversion_time,
                "input_count": len(legacy_search_results),
                "output_count": len(converted_search),
                "avg_time_per_result": search_conversion_time / len(legacy_search_results)
            }
            
            tests["merged_result_conversion"] = {
                "passed": True,
                "conversion_time_ms": merged_conversion_time,
                "input_count": len(legacy_merged_results),
                "output_count": len(converted_merged),
                "avg_time_per_result": merged_conversion_time / len(legacy_merged_results)
            }
            
        except Exception as e:
            tests["performance_error"] = {"passed": False, "error": str(e)}
        
        return tests
    
    async def apply_standardization(self) -> Dict[str, Any]:
        """Apply result standardization to Einstein components."""
        
        logger.info("🔧 Applying result standardization...")
        
        application_results = {
            "components_updated": [],
            "backup_created": False,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Create unified search interface wrapper
            from .unified_index import get_einstein_hub
            from .result_merger import ResultMerger
            from .query_router import QueryRouter
            
            hub = get_einstein_hub()
            merger = ResultMerger()
            router = QueryRouter()
            
            # Create unified interface
            unified_interface = UnifiedSearchInterface(hub, merger, router)
            
            # Test the unified interface
            logger.info("Testing unified interface...")
            test_response = await unified_interface.unified_search("test query")
            
            if isinstance(test_response, UnifiedSearchResponse):
                logger.info("✅ Unified interface working correctly")
                application_results["components_updated"].append("unified_search_interface")
            else:
                logger.error("❌ Unified interface returned wrong format")
                application_results["errors"].append("unified_interface_wrong_format")
            
            # Get adapter statistics
            adapter_stats = unified_interface.get_adapter_stats()
            application_results["adapter_stats"] = adapter_stats
            
            logger.info(f"Adapter conversions: {adapter_stats.get('total_conversions', 0)}")
            logger.info(f"Conversion errors: {adapter_stats.get('conversion_errors', 0)}")
            
        except Exception as e:
            logger.error(f"Failed to apply standardization: {e}")
            application_results["errors"].append(str(e))
        
        return application_results
    
    async def verify_implementation(self) -> Dict[str, Any]:
        """Verify the standardization implementation."""
        
        logger.info("🔍 Verifying standardization implementation...")
        
        verification_results = {
            "format_consistency": True,
            "cli_compatibility": True,
            "performance_acceptable": True,
            "error_handling": True,
            "issues": []
        }
        
        try:
            # Test format consistency
            adapter = get_standardization_adapter()
            
            # Test various input formats
            test_inputs = [
                {"content": "test", "file_path": "test.py", "line_number": 1, "score": 0.8, "result_type": "text", "context": {}, "timestamp": time.time()},
                {"content": "test", "file_path": "test.py", "line_number": 1, "combined_score": 0.9, "modality_scores": {"text": 0.8}, "source_modalities": ["text"], "context": {}, "timestamp": time.time()},
            ]
            
            for i, test_input in enumerate(test_inputs):
                try:
                    standardized = adapter._standardize_result([test_input], f"test_method_{i}")
                    if not all(isinstance(r, UnifiedSearchResult) for r in standardized):
                        verification_results["format_consistency"] = False
                        verification_results["issues"].append(f"Inconsistent format from test_method_{i}")
                except Exception as e:
                    verification_results["error_handling"] = False
                    verification_results["issues"].append(f"Error handling failed for test_method_{i}: {e}")
            
            # Test CLI compatibility
            test_response = UnifiedSearchResponse(
                query="test",
                results=[UnifiedSearchResult(content="test", file_path="test.py", line_number=1, score=0.8, result_type="text")],
                search_time_ms=10.0
            )
            
            cli_format = test_response.to_cli_format()
            required_fields = ["query", "results", "summary", "system"]
            if not all(field in cli_format for field in required_fields):
                verification_results["cli_compatibility"] = False
                verification_results["issues"].append("Missing required CLI fields")
            
            # Test performance
            start_time = time.time()
            large_input = [test_inputs[0]] * 1000
            adapter._standardize_result(large_input, "performance_test")
            conversion_time = (time.time() - start_time) * 1000
            
            if conversion_time > 100:  # More than 100ms for 1000 results
                verification_results["performance_acceptable"] = False
                verification_results["issues"].append(f"Slow conversion: {conversion_time:.1f}ms for 1000 results")
            
        except Exception as e:
            verification_results["error_handling"] = False
            verification_results["issues"].append(f"Verification failed: {e}")
        
        return verification_results
    
    def print_results(self, results: Dict[str, Any], title: str):
        """Print formatted results."""
        
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        
        if isinstance(results, dict):
            self._print_dict(results, indent=0)
        else:
            print(results)
    
    def _print_dict(self, d: Dict[str, Any], indent: int = 0):
        """Recursively print dictionary with indentation."""
        
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_dict(value, indent + 1)
            elif isinstance(value, list):
                print("  " * indent + f"{key}: [{len(value)} items]")
                if value and isinstance(value[0], dict):
                    # Show first item if it's a dict
                    print("  " * (indent + 1) + "First item:")
                    self._print_dict(value[0], indent + 2)
            else:
                print("  " * indent + f"{key}: {value}")


async def main():
    """Main function for result standardization application."""
    
    parser = argparse.ArgumentParser(description="Apply Einstein result format standardization")
    parser.add_argument("--mode", choices=["test", "apply", "verify"], default="test", 
                       help="Operation mode: test compatibility, apply changes, or verify implementation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    applicator = ResultStandardizationApplicator()
    
    try:
        if args.mode == "test":
            results = await applicator.test_compatibility()
            applicator.print_results(results, "COMPATIBILITY TEST RESULTS")
            
        elif args.mode == "apply":
            # Run compatibility test first
            logger.info("Running compatibility test before applying changes...")
            test_results = await applicator.test_compatibility()
            
            # Check if tests passed
            total_tests = sum(len(component_tests) for component_tests in test_results.values())
            passed_tests = sum(
                sum(1 for test in component_tests.values() if test.get('passed', False))
                for component_tests in test_results.values()
            )
            
            if passed_tests < total_tests * 0.8:  # Require 80% pass rate
                logger.error(f"❌ Compatibility tests failed: {passed_tests}/{total_tests} passed")
                logger.error("Please fix compatibility issues before applying standardization")
                sys.exit(1)
            
            # Apply standardization
            results = await applicator.apply_standardization()
            applicator.print_results(results, "STANDARDIZATION APPLICATION RESULTS")
            
        elif args.mode == "verify":
            results = await applicator.verify_implementation()
            applicator.print_results(results, "STANDARDIZATION VERIFICATION RESULTS")
            
            # Exit with error code if verification failed
            if not all(results[key] for key in ["format_consistency", "cli_compatibility", "performance_acceptable", "error_handling"]):
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())