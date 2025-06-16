"""
Task management for Bolt system.

Handles task ordering, dependencies, and scheduling.
"""

from collections import defaultdict, deque

from ..utils.logging import get_component_logger
from .types import Task


class TaskManager:
    """Manages task ordering and dependencies."""

    def __init__(self):
        self.logger = get_component_logger("task_manager")

    def order_tasks(self, tasks: list[Task]) -> list[Task]:
        """Order tasks based on dependencies and priority."""

        # First, resolve dependencies using topological sort
        dependency_ordered = self._topological_sort(tasks)

        # Then, within each level, sort by priority
        priority_ordered = self._sort_by_priority(dependency_ordered)

        return priority_ordered

    def _topological_sort(self, tasks: list[Task]) -> list[Task]:
        """Perform topological sort to handle dependencies."""

        # Build adjacency list and in-degree count
        task_map = {task.id: task for task in tasks}
        graph = defaultdict(list)
        in_degree = defaultdict(int)

        # Initialize in-degree for all tasks
        for task in tasks:
            in_degree[task.id] = 0

        # Build graph
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in task_map:
                    graph[dep_id].append(task.id)
                    in_degree[task.id] += 1
                else:
                    self.logger.warning(
                        f"Task {task.id} has unknown dependency: {dep_id}"
                    )

        # Kahn's algorithm for topological sorting
        queue: deque[str] = deque()
        result = []

        # Find tasks with no dependencies
        for task_id, degree in in_degree.items():
            if degree == 0:
                queue.append(task_id)

        while queue:
            current_id = queue.popleft()
            result.append(task_map[current_id])

            # Process neighbors
            for neighbor_id in graph[current_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)

        # Check for cycles
        if len(result) != len(tasks):
            cycle_tasks = [
                task_id for task_id, degree in in_degree.items() if degree > 0
            ]
            self.logger.error(
                f"Dependency cycle detected involving tasks: {cycle_tasks}"
            )
            # Return original order if cycle detected
            return tasks

        return result

    def _sort_by_priority(self, tasks: list[Task]) -> list[Task]:
        """Sort tasks by priority within dependency groups."""

        # Group tasks by their dependency level
        levels = self._group_by_dependency_level(tasks)

        # Sort each level by priority
        sorted_tasks = []
        for level in levels:
            # Sort by priority (higher priority first) and then by ID for consistency
            level_sorted = sorted(level, key=lambda t: (-t.priority.value, t.id))
            sorted_tasks.extend(level_sorted)

        return sorted_tasks

    def _group_by_dependency_level(self, tasks: list[Task]) -> list[list[Task]]:
        """Group tasks by their dependency level."""

        task_map = {task.id: task for task in tasks}
        levels = []
        processed: set[str] = set()

        while len(processed) < len(tasks):
            current_level = []

            for task in tasks:
                if task.id in processed:
                    continue

                # Check if all dependencies are processed
                deps_satisfied = all(
                    dep_id in processed or dep_id not in task_map
                    for dep_id in task.dependencies
                )

                if deps_satisfied:
                    current_level.append(task)

            if not current_level:
                # This shouldn't happen if topological sort worked correctly
                remaining_tasks = [task for task in tasks if task.id not in processed]
                self.logger.error(
                    f"Unable to resolve dependencies for tasks: "
                    f"{[task.id for task in remaining_tasks]}"
                )
                current_level = remaining_tasks  # Add remaining tasks to break the loop

            levels.append(current_level)
            processed.update(task.id for task in current_level)

        return levels

    def validate_dependencies(self, tasks: list[Task]) -> list[str]:
        """Validate task dependencies and return any errors."""

        errors = []
        task_ids = {task.id for task in tasks}

        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    errors.append(f"Task {task.id} has unknown dependency: {dep_id}")

                # Check for self-dependency
                if dep_id == task.id:
                    errors.append(f"Task {task.id} has self-dependency")

        # Check for dependency cycles
        if self._has_cycle(tasks):
            errors.append("Dependency cycle detected in task graph")

        return errors

    def _has_cycle(self, tasks: list[Task]) -> bool:
        """Check if task dependencies contain cycles."""

        task_map = {task.id: task for task in tasks}
        visited = set()
        rec_stack = set()

        def dfs(task_id: str) -> bool:
            if task_id in rec_stack:
                return True  # Cycle found
            if task_id in visited:
                return False  # Already processed

            visited.add(task_id)
            rec_stack.add(task_id)

            if task_id in task_map:
                for dep_id in task_map[task_id].dependencies:
                    if dfs(dep_id):
                        return True

            rec_stack.remove(task_id)
            return False

        return any(task.id not in visited and dfs(task.id) for task in tasks)

    def estimate_execution_time(self, tasks: list[Task]) -> float:
        """Estimate total execution time considering parallelization."""

        if not tasks:
            return 0.0

        # Group tasks by dependency level
        levels = self._group_by_dependency_level(tasks)

        total_time = 0.0

        for level in levels:
            # In each level, tasks can run in parallel
            # So the time is the maximum time of tasks in that level
            level_time = max(
                task.estimated_duration or 1.0  # Default to 1 second if not specified
                for task in level
            )
            total_time += level_time

        return total_time
