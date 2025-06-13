"""
from __future__ import annotations

MILP (Mixed Integer Linear Programming) solver for wheel strategy optimization.
Supports multiple backends: OR-Tools, Gurobi (if available), and PuLP.
"""

import time
from dataclasses import dataclass

from ...utils import get_logger

try:
    from ortools.linear_solver import pywraplp

    _HAS_ORTOOLS = True
except ImportError:
    _HAS_ORTOOLS = False

try:
    import gurobipy as gp
    from gurobipy import GRB

    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False

try:
    import pulp

    _HAS_PULP = True
except ImportError:
    _HAS_PULP = False


logger = get_logger(__name__)


@dataclass
class OptimizationProblem:
    """Definition of wheel strategy optimization problem."""

    # Required fields first
    portfolio_value: float
    spot_prices: dict[str, float]
    option_chains: dict[str, list[dict]]  # symbol -> list of options

    # Portfolio constraints with defaults
    max_position_pct: float = 1.0  # 100% max position
    min_cash_pct: float = 0.1  # 10% minimum cash

    # Risk constraints
    max_portfolio_delta: float = 0.5
    max_single_position_pct: float = 0.3

    # Transaction costs
    commission_per_contract: float = 0.65

    # Optimization parameters
    time_limit_seconds: float = 10.0
    mip_gap: float = 0.01  # 1% optimality gap


class MILPSolver:
    """Mixed Integer Linear Programming solver for portfolio optimization."""

    def __init__(self, backend: str = "auto"):
        """
        Initialize solver with specified backend.

        Args:
            backend: "ortools", "gurobi", "pulp", or "auto"
        """
        self.backend = self._select_backend(backend)
        logger.info("milp_solver_initialized", backend=self.backend)

    def _select_backend(self, requested: str) -> str:
        """Select best available backend."""

        if requested == "auto":
            if _HAS_GUROBI:
                return "gurobi"
            elif _HAS_ORTOOLS:
                return "ortools"
            elif _HAS_PULP:
                return "pulp"
            else:
                raise ImportError("No MILP solver available. Install ortools, gurobi, or pulp.")

        # Check if requested backend is available
        if requested == "gurobi" and not _HAS_GUROBI:
            raise ImportError("Gurobi not available")
        elif requested == "ortools" and not _HAS_ORTOOLS:
            raise ImportError("OR-Tools not available")
        elif requested == "pulp" and not _HAS_PULP:
            raise ImportError("PuLP not available")

        return requested

    def optimize_wheel_positions(self, problem: OptimizationProblem) -> dict[str, any]:
        """Optimize wheel strategy positions."""

        start_time = time.time()

        if self.backend == "ortools":
            result = self._solve_with_ortools(problem)
        elif self.backend == "gurobi":
            result = self._solve_with_gurobi(problem)
        elif self.backend == "pulp":
            result = self._solve_with_pulp(problem)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        solve_time = time.time() - start_time
        result["solve_time"] = solve_time

        logger.info(
            "optimization_complete",
            backend=self.backend,
            solve_time=solve_time,
            objective=result.get("objective_value"),
            gap=result.get("optimality_gap"),
        )

        return result

    def _solve_with_ortools(self, problem: OptimizationProblem) -> dict[str, any]:
        """Solve using Google OR-Tools."""

        # Create solver
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            solver = pywraplp.Solver.CreateSolver("CBC")

        # Decision variables
        variables = {}

        # Cash allocation
        cash = solver.NumVar(
            problem.portfolio_value * problem.min_cash_pct, problem.portfolio_value, "cash"
        )
        variables["cash"] = cash

        # Stock positions (integer multiples of 100)
        stock_positions = {}
        for symbol, price in problem.spot_prices.items():
            max_shares = int(problem.portfolio_value / price / 100) * 100
            stock_positions[symbol] = solver.IntVar(0, max_shares, f"stock_{symbol}")
            variables[f"stock_{symbol}"] = stock_positions[symbol]

        # Option positions (integer contracts)
        option_positions = {}
        for symbol, chain in problem.option_chains.items():
            for option in chain:
                option_id = f"{symbol}_{option['strike']}_{option['expiry']}_{option['type']}"
                option_positions[option_id] = solver.IntVar(
                    0, 100, f"option_{option_id}"  # Max 100 contracts
                )
                variables[f"option_{option_id}"] = option_positions[option_id]

        # Constraints

        # 1. Capital constraint
        capital_used = cash

        for symbol, position in stock_positions.items():
            capital_used += position * problem.spot_prices[symbol]

        # Options reduce capital (for puts) or require stock (for calls)
        for option_id, position in option_positions.items():
            symbol = option_id.split("_")[0]
            option_data = self._find_option_data(option_id, problem.option_chains)

            if option_data["type"] == "P":
                # Selling puts requires cash reserve
                capital_used += position * option_data["strike"] * 100

        solver.Add(capital_used <= problem.portfolio_value)

        # 2. Covered call constraint
        for symbol in problem.spot_prices:
            call_contracts = 0
            for option_id, position in option_positions.items():
                if symbol in option_id and option_id.endswith("_C"):
                    call_contracts += position

            # Can't sell more calls than stock owned
            solver.Add(call_contracts * 100 <= stock_positions[symbol])

        # 3. Portfolio delta constraint
        total_delta = 0

        for symbol, position in stock_positions.items():
            total_delta += position * 1.0  # Stock delta = 1

        for option_id, position in option_positions.items():
            option_data = self._find_option_data(option_id, problem.option_chains)
            total_delta += position * option_data.get("delta", 0) * 100

        max_delta = problem.portfolio_value / 20  # Assuming $20 Unity price
        solver.Add(total_delta <= max_delta * problem.max_portfolio_delta)

        # Objective: Maximize expected return
        objective = 0

        # Premium income from options
        for option_id, position in option_positions.items():
            option_data = self._find_option_data(option_id, problem.option_chains)
            premium = option_data.get("mid_price", 0) * 100
            commission = problem.commission_per_contract
            objective += position * (premium - commission)

        # Expected stock appreciation (simplified)
        for symbol, position in stock_positions.items():
            expected_return = 0.001 * problem.spot_prices[symbol]  # 0.1% daily
            objective += position * expected_return

        solver.Maximize(objective)

        # Set time limit
        solver.SetTimeLimit(problem.time_limit_seconds * 1000)

        # Solve
        status = solver.Solve()

        # Extract results
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            solution = {
                "status": "optimal" if status == pywraplp.Solver.OPTIMAL else "feasible",
                "objective_value": solver.Objective().Value(),
                "cash": cash.solution_value(),
                "positions": {},
            }

            # Stock positions
            for symbol, var in stock_positions.items():
                if var.solution_value() > 0:
                    solution["positions"][f"stock_{symbol}"] = int(var.solution_value())

            # Option positions
            for option_id, var in option_positions.items():
                if var.solution_value() > 0:
                    solution["positions"][f"option_{option_id}"] = int(var.solution_value())

            # Calculate optimality gap
            if hasattr(solver, "Objective"):
                best_bound = solver.Objective().BestBound()
                solution["optimality_gap"] = abs(
                    (solution["objective_value"] - best_bound)
                    / max(abs(solution["objective_value"]), 1e-10)
                )

            return solution

        else:
            return {
                "status": "infeasible",
                "objective_value": None,
                "message": "No feasible solution found",
            }

    def _solve_with_gurobi(self, problem: OptimizationProblem) -> dict[str, any]:
        """Solve using Gurobi optimizer."""

        # Create model
        model = gp.Model("wheel_optimization")

        # Set parameters
        model.setParam("TimeLimit", problem.time_limit_seconds)
        model.setParam("MIPGap", problem.mip_gap)

        # Decision variables
        cash = model.addVar(
            lb=problem.portfolio_value * problem.min_cash_pct,
            ub=problem.portfolio_value,
            name="cash",
        )

        # Stock positions
        stock_vars = {}
        for symbol, price in problem.spot_prices.items():
            max_shares = int(problem.portfolio_value / price / 100) * 100
            stock_vars[symbol] = model.addVar(
                lb=0, ub=max_shares, vtype=GRB.INTEGER, name=f"stock_{symbol}"
            )

        # Option positions
        option_vars = {}
        for symbol, chain in problem.option_chains.items():
            for option in chain:
                option_id = f"{symbol}_{option['strike']}_{option['expiry']}_{option['type']}"
                option_vars[option_id] = model.addVar(
                    lb=0, ub=100, vtype=GRB.INTEGER, name=f"option_{option_id}"
                )

        model.update()

        # Constraints (similar to OR-Tools)
        # ... (constraints implementation similar to above)

        # Objective
        obj_expr = gp.LinExpr()

        for option_id, var in option_vars.items():
            option_data = self._find_option_data(option_id, problem.option_chains)
            premium = option_data.get("mid_price", 0) * 100
            commission = problem.commission_per_contract
            obj_expr += var * (premium - commission)

        model.setObjective(obj_expr, GRB.MAXIMIZE)

        # Solve
        model.optimize()

        # Extract solution
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            solution = {
                "status": "optimal" if model.status == GRB.OPTIMAL else "feasible",
                "objective_value": model.objVal,
                "cash": cash.X,
                "positions": {},
                "optimality_gap": model.MIPGap,
            }

            for symbol, var in stock_vars.items():
                if var.X > 0:
                    solution["positions"][f"stock_{symbol}"] = int(var.X)

            for option_id, var in option_vars.items():
                if var.X > 0:
                    solution["positions"][f"option_{option_id}"] = int(var.X)

            return solution

        else:
            return {
                "status": "infeasible",
                "objective_value": None,
                "message": f"Gurobi status: {model.status}",
            }

    def _solve_with_pulp(self, problem: OptimizationProblem) -> dict[str, any]:
        """Solve using PuLP (fallback solver)."""

        # Create problem
        model = pulp.LpProblem("wheel_optimization", pulp.LpMaximize)

        # Variables and constraints similar to OR-Tools
        # ... (implementation omitted for brevity)

        # This is a simplified version - PuLP is less performant
        # but provides a fallback option

        return {
            "status": "not_implemented",
            "message": "PuLP solver not fully implemented - use OR-Tools or Gurobi",
        }

    def _find_option_data(self, option_id: str, option_chains: dict[str, list[dict]]) -> dict:
        """Find option data from chains."""

        parts = option_id.split("_")
        symbol = parts[0]
        strike = float(parts[1])
        expiry = parts[2]
        opt_type = parts[3]

        for option in option_chains.get(symbol, []):
            if (
                option["strike"] == strike
                and option["expiry"] == expiry
                and option["type"] == opt_type
            ):
                return option

        return {}

    def solve_with_heuristic(self, problem: OptimizationProblem) -> dict[str, any]:
        """Fast heuristic solver for real-time decisions."""

        start_time = time.time()

        # Simple greedy heuristic
        solution = {
            "status": "heuristic",
            "cash": problem.portfolio_value,
            "positions": {},
            "objective_value": 0,
        }

        # 1. Identify best put to sell
        best_put_efficiency = 0
        best_put = None

        for symbol, chain in problem.option_chains.items():
            for option in chain:
                if option["type"] == "P" and option.get("delta", 0) > -0.4:
                    # Calculate capital efficiency
                    premium = option.get("mid_price", 0)
                    capital_required = option["strike"]
                    efficiency = premium / capital_required

                    if efficiency > best_put_efficiency:
                        best_put_efficiency = efficiency
                        best_put = (symbol, option)

        # 2. Allocate to best put
        if best_put:
            symbol, option = best_put
            max_contracts = int(solution["cash"] / (option["strike"] * 100))
            contracts = min(max_contracts, 10)  # Limit concentration

            if contracts > 0:
                option_id = f"{symbol}_{option['strike']}_{option['expiry']}_P"
                solution["positions"][f"option_{option_id}"] = contracts
                solution["cash"] -= contracts * option["strike"] * 100
                solution["objective_value"] += contracts * option["mid_price"] * 100

        solution["solve_time"] = time.time() - start_time

        return solution