"""Parallel MCTS search workers for P-cores.

Runs Monte Carlo Tree Search in parallel across P-cores to explore
thousands of code implementations efficiently.
"""
import asyncio
import logging
import multiprocessing as mp
import queue
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchRequest:
    """Request for code search."""

    id: str
    query: str
    context: dict[str, Any]
    guidance: dict[str, np.ndarray]  # Neural guidance (value, policy)
    simulations: int = 2000
    exploration_constant: float = 1.414


@dataclass
class SearchResult:
    """Result from code search."""

    id: str
    best_code: str
    confidence: float
    alternatives: list[dict[str, Any]]
    search_tree: Optional["TreeNode"] = None
    stats: dict[str, Any] = None


class TreeNode:
    """Node in MCTS search tree."""

    def __init__(
        self,
        state: str,
        parent: Optional["TreeNode"] = None,
        action: str | None = None,
        prior: float = 1.0,
    ):
        self.state = state  # Current code state
        self.parent = parent
        self.action = action  # Action that led to this state
        self.prior = prior  # Prior probability from policy network

        self.visits = 0
        self.value_sum = 0.0
        self.children: dict[str, TreeNode] = {}
        self.is_expanded = False

    @property
    def value(self) -> float:
        """Average value of this node."""
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    @property
    def ucb_score(self) -> float:
        """Upper Confidence Bound score for selection."""
        if self.visits == 0:
            return float("inf")

        exploration = np.sqrt(2 * np.log(self.parent.visits) / self.visits)
        return self.value + 1.414 * exploration * self.prior

    def select_child(self) -> "TreeNode":
        """Select best child using UCB."""
        return max(self.children.values(), key=lambda n: n.ucb_score)

    def expand(self, actions: list[tuple[str, float]]) -> "TreeNode":
        """Expand node with possible actions."""
        self.is_expanded = True

        for action, prior in actions:
            if action not in self.children:
                self.children[action] = TreeNode(
                    state=self._apply_action(action),
                    parent=self,
                    action=action,
                    prior=prior,
                )

        # Return random child for exploration
        if self.children:
            return list(self.children.values())[0]
        return self

    def _apply_action(self, action: str) -> str:
        """Apply action to current state to get new state."""
        # Simplified: append action to code
        return f"{self.state}\n{action}"

    def backup(self, value: float):
        """Backup value through tree."""
        self.visits += 1
        self.value_sum += value

        if self.parent:
            self.parent.backup(value)


class CodeActionSpace:
    """Defines possible code transformations using real code generation."""

    # Action types for code transformation
    ACTION_TYPES = [
        "add_function",
        "add_class",
        "add_import",
        "modify_function",
        "add_type_hint",
        "add_docstring",
        "refactor_loop",
        "add_error_handling",
        "optimize_algorithm",
        "add_test",
    ]

    def __init__(self):
        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent))

        from core.code_generator import CodeGenerator
        from core.code_understanding import CodeAnalyzer, CodeTransformer

        self.generator = CodeGenerator()
        self.analyzer = CodeAnalyzer()
        self.transformer = CodeTransformer()
        self._query_cache = {}

    def set_query_context(self, query: str, context: dict[str, Any]):
        """Set the query context for code generation."""
        self._query_cache = {"query": query, "context": context}

    def get_actions(
        self, state: str, policy_probs: np.ndarray | None = None
    ) -> list[tuple[str, float]]:
        """Get possible actions from current state."""
        actions = []

        # Analyze current state
        code_context = self.analyzer.analyze(state)

        # Filter relevant actions based on state
        relevant_actions = self._filter_relevant_actions(state, code_context)

        # Use policy network probabilities if available
        if policy_probs is not None and len(policy_probs) >= len(self.ACTION_TYPES):
            for i, action_type in enumerate(self.ACTION_TYPES):
                if action_type in relevant_actions:
                    prob = float(policy_probs[i])
                    action = self._generate_action(action_type, state, code_context)
                    if action:
                        actions.append((action, prob))
        else:
            # Uniform priors for relevant actions
            for action_type in relevant_actions:
                action = self._generate_action(action_type, state, code_context)
                if action:
                    actions.append((action, 1.0 / len(relevant_actions)))

        return actions

    def _filter_relevant_actions(self, state: str, context) -> list[str]:
        """Filter actions based on current code state."""
        relevant = []

        # Always can add imports
        relevant.append("add_import")

        # If no functions, prioritize adding functions
        if not context.functions:
            relevant.extend(["add_function", "add_class"])
        else:
            # Can modify existing code
            relevant.extend(
                [
                    "modify_function",
                    "add_type_hint",
                    "add_docstring",
                    "add_error_handling",
                    "optimize_algorithm",
                ]
            )

        # Can add tests if there's code to test
        if context.functions or context.classes:
            relevant.append("add_test")

        # Can refactor if there's complexity
        if context.complexity > 5:
            relevant.append("refactor_loop")

        return relevant

    def _generate_action(self, action_type: str, state: str, context) -> str:
        """Generate specific action based on type using real code generation."""
        query = self._query_cache.get("query", "")

        try:
            if action_type == "add_function":
                # Generate function based on query
                from core.code_generator import GenerationRequest

                req = GenerationRequest(
                    query=query or "add a utility function",
                    context={"existing_functions": context.functions},
                    existing_code=state,
                )
                new_code = self.generator.generate(req)
                # Extract just the new function
                return self._extract_new_function(state, new_code)

            elif action_type == "add_class":
                # Generate class based on query
                class_query = (
                    f"create a class for {query}" if query else "create a utility class"
                )
                from core.code_generator import GenerationRequest

                req = GenerationRequest(
                    query=class_query,
                    context={"existing_classes": context.classes},
                    existing_code=state,
                )
                new_code = self.generator.generate(req)
                return self._extract_new_class(state, new_code)

            elif action_type == "add_import":
                # Add relevant import based on code needs
                import_needed = self._infer_needed_import(state, context)
                if import_needed:
                    return self.transformer.add_import(state, import_needed)
                return None

            elif action_type == "add_type_hint":
                # Add type hints to functions
                if context.functions and not context.type_hints:
                    return self.transformer.add_type_hints(state)
                return None

            elif action_type == "add_docstring":
                # Add docstrings to undocumented functions
                for func in context.functions:
                    if not func.get("docstring"):
                        # Add docstring for first undocumented function
                        return self._add_function_docstring(state, func["name"])
                return None

            elif action_type == "add_error_handling":
                # Add error handling to functions
                for func in context.functions:
                    if not self._has_error_handling(state, func["name"]):
                        return self.transformer.add_error_handling(state, func["name"])
                return None

            elif action_type == "optimize_algorithm":
                # Optimize existing code
                from core.code_generator import GenerationRequest

                req = GenerationRequest(
                    query=f"optimize this code: {state[:200]}",
                    context={},
                    existing_code=state,
                )
                return self.generator._optimize_code("", context, req)

            elif action_type == "add_test":
                # Generate test for existing function
                if context.functions:
                    func = context.functions[0]
                    from core.code_generator import GenerationRequest

                    req = GenerationRequest(
                        query=f"create test for {func['name']}",
                        context={"function": func},
                        existing_code=state,
                    )
                    return self.generator._generate_test(func["name"], context, req)
                return None

            elif action_type == "refactor_loop":
                # Refactor loops for efficiency
                return self._refactor_loops(state, context)

            else:
                return None

        except Exception as e:
            # Fallback to simple generation
            return f"# {action_type}: {str(e)[:50]}"

    def _extract_new_function(self, old_code: str, new_code: str) -> str:
        """Extract only the new function from generated code."""
        # Simple approach: find what's different
        old_lines = set(old_code.strip().split("\n"))
        new_lines = new_code.strip().split("\n")

        func_lines = []
        in_function = False

        for line in new_lines:
            if line not in old_lines:
                if line.strip().startswith("def "):
                    in_function = True
                if in_function:
                    func_lines.append(line)
                    if (
                        line.strip()
                        and not line.startswith(" ")
                        and not line.startswith("\t")
                    ):
                        if not line.strip().startswith("def"):
                            break

        return "\n".join(func_lines) if func_lines else new_code

    def _extract_new_class(self, old_code: str, new_code: str) -> str:
        """Extract only the new class from generated code."""
        old_lines = set(old_code.strip().split("\n"))
        new_lines = new_code.strip().split("\n")

        class_lines = []
        in_class = False
        indent_level = 0

        for line in new_lines:
            if line not in old_lines:
                if line.strip().startswith("class "):
                    in_class = True
                    indent_level = len(line) - len(line.lstrip())
                if in_class:
                    class_lines.append(line)
                    # Check if we've left the class
                    if line.strip() and len(line) - len(line.lstrip()) <= indent_level:
                        if not line.strip().startswith("class"):
                            break

        return "\n".join(class_lines) if class_lines else new_code

    def _infer_needed_import(self, state: str, context) -> str | None:
        """Infer what import might be needed."""
        code_lower = state.lower()

        # Check for common patterns that need imports
        if "array" in code_lower or "matrix" in code_lower:
            if "numpy" not in str(context.imports):
                return "numpy"

        if "dataframe" in code_lower or "series" in code_lower:
            if "pandas" not in str(context.imports):
                return "pandas"

        if "plot" in code_lower or "graph" in code_lower:
            if "matplotlib" not in str(context.imports):
                return "matplotlib.pyplot"

        if any(t in code_lower for t in ["list[", "dict[", "optional[", "union["]):
            if "typing" not in str(context.imports):
                return "typing"

        return None

    def _add_function_docstring(self, code: str, func_name: str) -> str:
        """Add docstring to function."""
        import ast

        import astor

        tree = ast.parse(code)

        class DocstringAdder(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if node.name == func_name:
                    # Generate appropriate docstring
                    params = [arg.arg for arg in node.args.args if arg.arg != "self"]
                    docstring = f'"""Function {func_name}.\n\n'
                    if params:
                        docstring += "Args:\n"
                        for param in params:
                            docstring += f"    {param}: Description\n"
                    docstring += '"""'

                    # Add docstring as first statement
                    doc_node = ast.Expr(value=ast.Constant(value=docstring))
                    node.body.insert(0, doc_node)

                return node

        transformer = DocstringAdder()
        new_tree = transformer.visit(tree)

        return astor.to_source(new_tree)

    def _has_error_handling(self, code: str, func_name: str) -> bool:
        """Check if function has error handling."""
        # Simple check - could be more sophisticated
        lines = code.split("\n")
        in_function = False

        for line in lines:
            if f"def {func_name}" in line:
                in_function = True
            elif in_function and line.strip() and not line.startswith(" "):
                break
            elif in_function and "try:" in line:
                return True

        return False

    def _refactor_loops(self, code: str, context) -> str:
        """Refactor loops for efficiency."""
        # Look for common patterns that can be optimized
        optimized = code

        # Replace simple loops with comprehensions
        import re

        # Pattern: for loop that appends
        pattern = r"(\w+)\s*=\s*\[\]\s*\n\s*for\s+(\w+)\s+in\s+(\w+):\s*\n\s*\1\.append\(([^)]+)\)"
        replacement = r"\1 = [\4 for \2 in \3]"
        optimized = re.sub(pattern, replacement, optimized)

        return optimized


class MCTSSearcher:
    """Monte Carlo Tree Search for code generation."""

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.action_space = CodeActionSpace()

    def search(self, request: SearchRequest) -> SearchResult:
        """Run MCTS to find best code."""
        start_time = time.perf_counter()

        # Initialize root with query as initial state
        root = TreeNode(state=f"# Solution for: {request.query}")

        # Set query context for code generation
        self.action_space.set_query_context(request.query, request.context)

        # Get policy guidance if available
        policy_probs = request.guidance.get("policy")
        if policy_probs is not None:
            policy_probs = policy_probs.flatten()

        # Run simulations
        for _sim in range(request.simulations):
            node = root

            # Selection: traverse tree using UCB
            path = [node]
            while node.is_expanded and node.children:
                node = node.select_child()
                path.append(node)

            # Expansion: add new nodes
            if not node.is_expanded and node.visits > 0:
                actions = self.action_space.get_actions(node.state, policy_probs)
                node = node.expand(actions)
                path.append(node)

            # Evaluation: use value network or rollout
            value = self._evaluate(node, request.guidance.get("value"))

            # Backup: propagate value up the tree
            node.backup(value)

        # Extract best path
        best_code, confidence = self._extract_best_solution(root)
        alternatives = self._extract_alternatives(root, n=3)

        elapsed = time.perf_counter() - start_time

        return SearchResult(
            id=request.id,
            best_code=best_code,
            confidence=confidence,
            alternatives=alternatives,
            search_tree=root,
            stats={
                "simulations": request.simulations,
                "worker_id": self.worker_id,
                "search_time_ms": elapsed * 1000,
                "nodes_explored": self._count_nodes(root),
            },
        )

    def _evaluate(self, node: TreeNode, value_guidance: np.ndarray | None) -> float:
        """Evaluate a node's value using real code quality metrics."""
        # Use neural value if available
        if value_guidance is not None and value_guidance.size > 0:
            return float(value_guidance[0])

        # Use real code analysis
        code = node.state

        # Import analyzer here
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from core.code_understanding import CodeAnalyzer, CodeValidator

        if not hasattr(self, "_analyzer"):
            self._analyzer = CodeAnalyzer()
            self._validator = CodeValidator()

        # Analyze code quality
        score = 0.5  # Base score

        try:
            # Check syntax validity
            is_valid, error = self._validator.validate_syntax(code)
            if not is_valid:
                return 0.1  # Low score for invalid syntax

            # Analyze code structure
            context = self._analyzer.analyze(code)

            # Reward complete solutions
            if context.functions:
                score += 0.15 * min(len(context.functions), 3)  # Up to 3 functions
            if context.classes:
                score += 0.1 * min(len(context.classes), 2)  # Up to 2 classes

            # Reward good practices
            if context.docstring:
                score += 0.05
            if context.type_hints:
                score += 0.05

            # Check complexity (lower is better)
            if context.complexity <= 5:
                score += 0.1
            elif context.complexity > 10:
                score -= 0.1

            # Reward imports (indicates using libraries)
            if context.imports:
                score += 0.05

            # Check for tests
            if any("test" in func["name"].lower() for func in context.functions):
                score += 0.1

            # Check for error handling
            if "try" in code and "except" in code:
                score += 0.05

        except Exception:
            # If analysis fails, use simple heuristics
            if "def" in code or "class" in code:
                score += 0.1
            if "return" in code:
                score += 0.1

        return max(0.0, min(1.0, score))

    def _extract_best_solution(self, root: TreeNode) -> tuple[str, float]:
        """Extract best code path from tree."""
        # Follow most visited path
        node = root
        code_parts = [node.state]

        while node.children:
            # Choose most visited child
            best_child = max(node.children.values(), key=lambda n: n.visits)
            if best_child.action:
                code_parts.append(best_child.action)
            node = best_child

        best_code = "\n".join(code_parts)
        confidence = node.value if node.visits > 0 else 0.5

        return best_code, confidence

    def _extract_alternatives(self, root: TreeNode, n: int = 3) -> list[dict[str, Any]]:
        """Extract top N alternative solutions."""
        alternatives = []

        # Get top nodes by value
        all_nodes = []
        self._collect_nodes(root, all_nodes)

        # Sort by value * visits (quality * confidence)
        sorted_nodes = sorted(
            all_nodes, key=lambda n: n.value * np.sqrt(n.visits), reverse=True
        )

        for node in sorted_nodes[1 : n + 1]:  # Skip root
            if node.visits > 10:  # Minimum visits threshold
                path = self._get_path_to_node(node)
                code = "\n".join(
                    n.state if n == root else n.action
                    for n in path
                    if n.action or n == root
                )

                alternatives.append(
                    {"code": code, "confidence": node.value, "visits": node.visits}
                )

        return alternatives

    def _collect_nodes(self, node: TreeNode, nodes: list[TreeNode]):
        """Recursively collect all nodes."""
        nodes.append(node)
        for child in node.children.values():
            self._collect_nodes(child, nodes)

    def _get_path_to_node(self, node: TreeNode) -> list[TreeNode]:
        """Get path from root to node."""
        path = []
        while node:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def _count_nodes(self, node: TreeNode) -> int:
        """Count total nodes in tree."""
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count


class SearchWorkerProcess:
    """Process running MCTS search."""

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.process = None

        # Communication
        self.request_queue = mp.Queue(maxsize=10)
        self.response_queue = mp.Queue(maxsize=10)
        self.shutdown_event = mp.Event()

    def start(self):
        """Start worker process."""
        self.process = mp.Process(
            target=self._run_worker,
            args=(
                self.worker_id,
                self.request_queue,
                self.response_queue,
                self.shutdown_event,
            ),
            daemon=True,
        )
        self.process.start()
        logger.info(f"Search worker {self.worker_id} started")

    def stop(self):
        """Stop worker process."""
        self.shutdown_event.set()
        if self.process:
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
        logger.info(f"Search worker {self.worker_id} stopped")

    @staticmethod
    def _run_worker(
        worker_id: int,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        shutdown_event: mp.Event,
    ):
        """Main worker loop."""
        searcher = MCTSSearcher(worker_id)

        while not shutdown_event.is_set():
            try:
                # Get request
                request = request_queue.get(timeout=0.1)

                # Run search
                result = searcher.search(request)

                # Send result
                response_queue.put(result)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Search worker {worker_id} error: {e}")
                if "request" in locals():
                    response_queue.put(
                        SearchResult(
                            id=request.id,
                            best_code="# Error during search",
                            confidence=0.0,
                            alternatives=[],
                            stats={"error": str(e)},
                        )
                    )


class SearchWorkerPool:
    """Pool of search workers running on P-cores."""

    def __init__(self, num_workers: int = 8):
        self.num_workers = num_workers
        self.workers = []

        # Start workers
        for i in range(num_workers):
            worker = SearchWorkerProcess(i)
            worker.start()
            self.workers.append(worker)

    async def parallel_search(
        self,
        query: str,
        context: dict[str, Any],
        guidance: dict[str, np.ndarray],
        simulations: int = 2000,
    ) -> dict[str, Any]:
        """Run parallel MCTS search."""
        # Divide simulations across workers
        sims_per_worker = simulations // self.num_workers
        remainder = simulations % self.num_workers

        # Create requests
        requests = []
        for i, worker in enumerate(self.workers):
            worker_sims = sims_per_worker + (1 if i < remainder else 0)
            request = SearchRequest(
                id=f"{uuid.uuid4()}-w{i}",
                query=query,
                context=context,
                guidance=guidance,
                simulations=worker_sims,
            )
            requests.append((worker, request))

        # Submit all requests
        for worker, request in requests:
            worker.request_queue.put(request)

        # Gather results with timeout
        results = []
        pending_requests = {req.id: (worker, req) for worker, req in requests}
        start_time = time.time()
        timeout = 30.0  # 30 second timeout

        while pending_requests and (time.time() - start_time) < timeout:
            # Check all workers for any results
            for _req_id, (worker, request) in list(pending_requests.items()):
                try:
                    result = worker.response_queue.get_nowait()

                    # Check if this result matches any pending request
                    if result.id in pending_requests:
                        results.append(result)
                        del pending_requests[result.id]
                    else:
                        # Not one of our results, put it back
                        worker.response_queue.put(result)

                except queue.Empty:
                    continue

            # Brief sleep to avoid busy waiting
            if pending_requests:
                await asyncio.sleep(0.01)

        # Check if we got all results
        if pending_requests:
            logger.warning(
                f"Timeout waiting for {len(pending_requests)} search results"
            )

        # Combine results
        return self._combine_results(results)

    def _combine_results(self, results: list[SearchResult]) -> dict[str, Any]:
        """Combine results from parallel searches."""
        # Select best overall solution
        best_result = max(results, key=lambda r: r.confidence)

        # Aggregate alternatives
        all_alternatives = []
        for result in results:
            all_alternatives.extend(result.alternatives)

        # Sort and deduplicate
        unique_alternatives = {}
        for alt in all_alternatives:
            code_hash = hash(alt["code"])
            if (
                code_hash not in unique_alternatives
                or alt["confidence"] > unique_alternatives[code_hash]["confidence"]
            ):
                unique_alternatives[code_hash] = alt

        alternatives = sorted(
            unique_alternatives.values(), key=lambda a: a["confidence"], reverse=True
        )[:5]

        # Aggregate stats
        total_nodes = sum(r.stats.get("nodes_explored", 0) for r in results)
        avg_time = sum(r.stats.get("search_time_ms", 0) for r in results) / len(results)

        return {
            "best_code": best_result.best_code,
            "confidence": best_result.confidence,
            "alternatives": alternatives,
            "search_tree": best_result.search_tree,
            "stats": {
                "total_simulations": sum(
                    r.stats.get("simulations", 0) for r in results
                ),
                "total_nodes_explored": total_nodes,
                "avg_search_time_ms": avg_time,
                "num_workers": len(results),
            },
        }

    def shutdown(self):
        """Shutdown all workers."""
        for worker in self.workers:
            worker.stop()
