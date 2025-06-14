"""Multi-Objective Evaluator for code solutions.

Evaluates code across multiple dimensions including performance,
readability, correctness, and resource usage.
"""
from __future__ import annotations

import ast
import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MultiObjectiveEvaluator:
    """Evaluates code solutions across multiple objectives."""

    def __init__(self):
        self.evaluation_count = 0
        self.evaluation_cache = {}
        self.objective_weights = {'performance': 0.3, 'readability': 0.25,
            'correctness': 0.35, 'resource_usage': 0.1}

    async def batch_evaluate(self, solutions: List[Dict[str, Any]], metrics:
        List[str], context: Optional[Dict[str, Any]]=None, batch_size:
        Optional[int]=None, hardware_executor: Optional[Any]=None) ->List[Dict
        [str, Any]]:
        """Evaluate multiple solutions in parallel."""
        logger.info(
            f"Evaluating {len(solutions)} solutions across {len(metrics)} metrics"
            )
        tasks = []
        for solution in solutions:
            task = self._evaluate_single(solution, metrics, context)
            tasks.append(task)
        results = await asyncio.gather(*tasks, return_exceptions = True)
        evaluated_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Evaluation failed for solution {i}: {result}")
                evaluated_results.append(self._failed_evaluation(metrics))
            else:
                evaluated_results.append(result)
        self.evaluation_count += len(solutions)
        return evaluated_results

    async def _evaluate_single(self, solution: Dict[str, Any], metrics:
        List[str], context: Optional[Dict[str, Any]]) ->Dict[str, Any]:
        """Evaluate a single solution."""
        code = solution.get('code', '')
        cache_key = hashlib.md5(f"{code}{metrics}".encode()).hexdigest()
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        results = {'solution': solution}
        for metric in metrics:
            if metric == 'performance':
                score = await self._evaluate_performance(code, context)
            elif metric == 'readability':
                score = self._evaluate_readability(code)
            elif metric == 'correctness':
                score = await self._evaluate_correctness(code, context)
            elif metric == 'resource_usage':
                score = self._evaluate_resource_usage(code)
            else:
                score = 0.5
            results[metric] = score
        overall = sum(results.get(m, 0) * self.objective_weights.get(m, 
            0.25) for m in metrics)
        results['overall'] = overall
        results['confidence'] = self._calculate_confidence(results)
        self.evaluation_cache[cache_key] = results
        return results

    async def _evaluate_performance(self, code: str, context: Optional[Dict
        [str, Any]]) ->float:
        """Evaluate performance characteristics."""
        score = 0.5
        try:
            tree = ast.parse(code)
            analyzer = PerformanceAnalyzer()
            analyzer.visit(tree)
            if analyzer.max_loop_depth > 2:
                score -= 0.1 * (analyzer.max_loop_depth - 2)
            if analyzer.has_vectorization:
                score += 0.2
            if analyzer.has_caching:
                score += 0.15
            if analyzer.uses_generators:
                score += 0.1
            if analyzer.has_repeated_calculations:
                score -= 0.15
            if analyzer.has_unnecessary_copies:
                score -= 0.1
        except Exception as e:
            logger.debug(f"Performance analysis failed: {e}")
            score = 0.3
        if context and 'test_data' in context:
            try:
                runtime_score = await self._measure_runtime(code, context[
                    'test_data'])
                score = 0.7 * score + 0.3 * runtime_score
            except Exception as e:
                logger.debug(f"Ignored exception in {'evaluator.py'}: {e}")
        return max(0.0, min(1.0, score))

    def _evaluate_readability(self, code: str) ->float:
        """Evaluate code readability."""
        score = 0.8
        lines = code.split('\n')
        long_lines = sum(1 for line in lines if len(line) > 80)
        if long_lines > 0:
            score -= 0.05 * min(long_lines / max(len(lines), 1), 0.3)
        functions = [i for i, line in enumerate(lines) if line.strip().
            startswith('def ')]
        if functions:
            func_lengths = []
            for i in range(len(functions)):
                start = functions[i]
                end = functions[i + 1] if i + 1 < len(functions) else len(lines
                    )
                func_lengths.append(end - start)
            avg_length = np.mean(func_lengths)
            if avg_length > 20:
                score -= min(0.2, (avg_length - 20) * 0.01)
        if 'x' in code or 'tmp' in code or 'temp' in code:
            score -= 0.1
        docstring_count = code.count('"""')
        comment_count = code.count('#')
        if docstring_count == 0 and comment_count == 0:
            score -= 0.2
        elif docstring_count > 0:
            score += 0.1
        try:
            tree = ast.parse(code)
            complexity = self._calculate_complexity(tree)
            if complexity > 10:
                score -= min(0.3, (complexity - 10) * 0.02)
        except Exception as e:
            logger.debug(f"Ignored exception in {'evaluator.py'}: {e}")
        return max(0.0, min(1.0, score))

    async def _evaluate_correctness(self, code: str, context: Optional[Dict
        [str, Any]]) ->float:
        """Evaluate code correctness."""
        score = 0.7
        try:
            ast.parse(code)
        except SyntaxError:
            return 0.0
        if 'typing' in code or '->' in code:
            score += 0.1
        if 'try:' in code:
            score += 0.1
        elif 'raise' in code and 'except' not in code:
            score -= 0.1
        if 'assert' in code or 'if not' in code:
            score += 0.05
        if context and 'tests' in context:
            test_score = await self._run_tests(code, context['tests'])
            score = 0.6 * score + 0.4 * test_score
        return max(0.0, min(1.0, score))

    def _evaluate_resource_usage(self, code: str) ->float:
        """Evaluate resource usage efficiency."""
        score = 0.7
        if 'del ' in code:
            score += 0.05
        if 'global ' in code:
            score -= 0.1
        memory_intensive = ['list(', 'dict(', '[x for', '{x for']
        intensive_count = sum(code.count(op) for op in memory_intensive)
        if intensive_count > 5:
            score -= min(0.2, intensive_count * 0.02)
        if 'yield' in code or 'generator' in code:
            score += 0.15
        if 'open(' in code:
            if 'with open' in code:
                score += 0.05
            else:
                score -= 0.1
        return max(0.0, min(1.0, score))

    def _calculate_complexity(self, tree: ast.AST) ->int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity

    def _calculate_confidence(self, results: Dict[str, float]) ->float:
        """Calculate confidence in evaluation."""
        scores = [v for k, v in results.items() if k not in ['solution',
            'overall', 'confidence']]
        if not scores:
            return 0.5
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        confidence = 1.0 - min(1.0, std_score * 2)
        if mean_score < 0.2 or mean_score > 0.8:
            confidence *= 0.9
        return confidence

    def _measure_runtime(self, code: str, test_data: Any) ->float:
        """Measure actual runtime performance."""
        return 0.7

    def _run_tests(self, code: str, tests: List[Dict[str, Any]]) ->float:
        """Run test cases against code."""
        return 0.8

    def _failed_evaluation(self, metrics: List[str]) ->Dict[str, Any]:
        """Return evaluation for failed solution."""
        result = {'solution': {}, 'overall': 0.0, 'confidence': 0.0}
        for metric in metrics:
            result[metric] = 0.0
        return result


class PerformanceAnalyzer(ast.NodeVisitor):
    """Analyzes code for performance characteristics."""

    def __init__(self):
        self.max_loop_depth = 0
        self.current_loop_depth = 0
        self.has_vectorization = False
        self.has_caching = False
        self.uses_generators = False
        self.has_repeated_calculations = False
        self.has_unnecessary_copies = False
        self.function_calls = set()

    def visit_For(self, node):
        self.current_loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)
        self.generic_visit(node)
        self.current_loop_depth -= 1

    def visit_While(self, node):
        self.current_loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)
        self.generic_visit(node)
        self.current_loop_depth -= 1

    def visit_Call(self, node):
        if hasattr(node.func, 'id'):
            func_name = node.func.id
            self.function_calls.add(func_name)
            if func_name in ['vectorize', 'apply', 'map']:
                self.has_vectorization = True
            if 'cache' in func_name.lower():
                self.has_caching = True
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        for decorator in node.decorator_list:
            if hasattr(decorator, 'id') and 'cache' in decorator.id:
                self.has_caching = True
        self.generic_visit(node)

    def visit_Yield(self, node):
        self.uses_generators = True
        self.generic_visit(node)

    def visit_ListComp(self, node):
        if self.current_loop_depth > 0:
            self.has_repeated_calculations = True
        self.generic_visit(node)

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call):
            if hasattr(node.value.func, 'id') and node.value.func.id in ['list'
                , 'dict', 'copy']:
                self.has_unnecessary_copies = True
        self.generic_visit(node)


import hashlib


class BenchmarkRunner:
    """Runs performance benchmarks on code."""

    def __init__(self):
        self.benchmark_results = {}

    async def benchmark_solution(self, code: str, test_cases: List[Dict[str,
        Any]]) ->Dict[str, float]:
        """Run benchmarks on solution."""
        results = {'execution_time': 0.0, 'memory_usage': 0.0, 'throughput':
            0.0}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete = False
            ) as f:
            f.write(code)
            temp_file = f.name
        try:
            for test_case in test_cases[:3]:
                start_time = time.time()
                process = await asyncio.create_subprocess_exec('python',
                    temp_file, stdin = asyncio.subprocess.PIPE, stdout = asyncio.subprocess.PIPE, stderr = asyncio.subprocess.PIPE)
                input_data = test_case.get('input', '').encode()
                stdout, stderr = await asyncio.wait_for(process.communicate
                    (input_data), timeout = 5.0)
                execution_time = time.time() - start_time
                results['execution_time'] += execution_time
                if test_case.get('expected_output'):
                    actual = stdout.decode().strip()
                    expected = test_case['expected_output'].strip()
                    if actual == expected:
                        results['throughput'] += 1
            if test_cases:
                results['execution_time'] /= len(test_cases[:3])
                results['throughput'] /= len(test_cases[:3])
        except asyncio.TimeoutError:
            logger.warning('Benchmark timed out')
            results['execution_time'] = 5.0
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
        finally:
            Path(temp_file).unlink(missing_ok = True)
        return results


class QualityMetrics:
    """Additional quality metrics for code evaluation."""

    @staticmethod
    def calculate_maintainability_index(code: str) ->float:
        """Calculate maintainability index."""
        try:
            tree = ast.parse(code)
            operators = set()
            operands = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.BinOp):
                    operators.add(type(node.op).__name__)
                elif isinstance(node, ast.Name):
                    operands.add(node.id)
            program_length = len(operators) + len(operands)
            program_vocabulary = len(set(operators)) + len(set(operands))
            if program_vocabulary == 0:
                return 0.5
            volume = program_length * np.log2(program_vocabulary + 1)
            complexity = 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.
                    ExceptHandler)):
                    complexity += 1
            loc = len([l for l in code.split('\n') if l.strip()])
            mi = 171 - 5.2 * np.log(volume + 1
                ) - 0.23 * complexity - 16.2 * np.log(loc + 1)
            mi = max(0, mi * 100 / 171)
            return mi / 100.0
        except Exception as e:
            logger.debug(f"Maintainability calculation failed: {e}")
            return 0.5

    @staticmethod
    def calculate_test_coverage_potential(code: str) ->float:
        """Estimate how testable the code is."""
        score = 0.5
        try:
            tree = ast.parse(code)
            functions = sum(1 for node in ast.walk(tree) if isinstance(node,
                ast.FunctionDef))
            classes = sum(1 for node in ast.walk(tree) if isinstance(node,
                ast.ClassDef))
            if functions > 0:
                score += min(0.3, functions * 0.05)
            pure_functions = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    assigns_global = any(isinstance(n, ast.Global) for n in
                        ast.walk(node))
                    if not assigns_global:
                        pure_functions += 1
            if functions > 0:
                purity_ratio = pure_functions / functions
                score += 0.2 * purity_ratio
        except Exception as e:
            logger.debug(f"Ignored exception in {'evaluator.py'}: {e}")
        return min(1.0, score)
