#!/usr/bin/env python3
"""Opik trace MCP server for LLM observability."""

from mcp.server import FastMCP
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os

mcp = FastMCP("trace-opik")

# Opik configuration
OPIK_BASE_URL = os.environ.get("OPIK_BASE_URL", "http://localhost:5173")
OPIK_API_URL = f"{OPIK_BASE_URL}/api"

@mcp.tool()
def log_llm_trace(
    operation: str,
    model: str = "claude-3.5",
    input_tokens: int = 0,
    output_tokens: int = 0,
    duration_ms: int = 0,
    status: str = "success",
    metadata: Optional[Dict] = None
) -> str:
    """Log an LLM operation trace to Opik.
    
    Args:
        operation: Name of the operation (e.g., "generate_recommendation")
        model: Model used (default: claude-3.5)
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        duration_ms: Operation duration in milliseconds
        status: Operation status (success/error)
        metadata: Additional metadata as JSON string
    """
    try:
        trace_data = {
            "name": operation,
            "type": "llm",
            "model": model,
            "start_time": datetime.now().isoformat(),
            "end_time": (datetime.now() + timedelta(milliseconds=duration_ms)).isoformat(),
            "metadata": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "duration_ms": duration_ms,
                "status": status
            }
        }
        
        if metadata:
            trace_data["metadata"].update(metadata)
        
        response = requests.post(
            f"{OPIK_API_URL}/traces",
            json=trace_data,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        trace_id = response.json().get("id", "unknown")
        return f"Logged LLM trace: {operation} (ID: {trace_id}, Tokens: {input_tokens}→{output_tokens})"
        
    except Exception as e:
        return f"Error logging LLM trace: {str(e)}"

@mcp.tool()
def log_evaluation_result(
    experiment_name: str,
    model: str,
    metrics: Dict[str, float],
    parameters: Optional[Dict] = None
) -> str:
    """Log model evaluation results to Opik.
    
    Args:
        experiment_name: Name of the experiment
        model: Model being evaluated
        metrics: Dictionary of metric names to values
        parameters: Model parameters used
    """
    try:
        eval_data = {
            "experiment": experiment_name,
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "parameters": parameters or {}
        }
        
        response = requests.post(
            f"{OPIK_API_URL}/evaluations",
            json=eval_data,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        eval_id = response.json().get("id", "unknown")
        
        # Format metrics for display
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        return f"Logged evaluation: {experiment_name} (ID: {eval_id})\nMetrics: {metrics_str}"
        
    except Exception as e:
        return f"Error logging evaluation: {str(e)}"

@mcp.tool()
def get_experiment_history(experiment_name: str, limit: int = 10) -> str:
    """Get history of an experiment from Opik.
    
    Args:
        experiment_name: Name of the experiment
        limit: Maximum number of results to return
    """
    try:
        params = {
            "experiment": experiment_name,
            "limit": limit,
            "sort": "timestamp_desc"
        }
        
        response = requests.get(f"{OPIK_API_URL}/evaluations", params=params)
        response.raise_for_status()
        
        evaluations = response.json()
        
        if not evaluations:
            return f"No history found for experiment: {experiment_name}"
        
        result = f"Experiment history for '{experiment_name}':\n\n"
        
        for eval in evaluations:
            timestamp = eval.get("timestamp", "")
            model = eval.get("model", "unknown")
            metrics = eval.get("metrics", {})
            
            result += f"• {timestamp} - {model}\n"
            for metric, value in metrics.items():
                result += f"  {metric}: {value:.4f}\n"
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"Error fetching experiment history: {str(e)}"

@mcp.tool()
def compare_models(
    experiment_name: str,
    metric: str,
    models: Optional[List[str]] = None
) -> str:
    """Compare model performance across runs.
    
    Args:
        experiment_name: Name of the experiment
        metric: Metric to compare (e.g., "accuracy", "f1_score")
        models: List of model names to compare (optional)
    """
    try:
        params = {
            "experiment": experiment_name,
            "limit": 100
        }
        
        response = requests.get(f"{OPIK_API_URL}/evaluations", params=params)
        response.raise_for_status()
        
        evaluations = response.json()
        
        if not evaluations:
            return f"No data found for experiment: {experiment_name}"
        
        # Group by model
        model_metrics = {}
        for eval in evaluations:
            model = eval.get("model", "unknown")
            if models and model not in models:
                continue
                
            metrics = eval.get("metrics", {})
            if metric in metrics:
                if model not in model_metrics:
                    model_metrics[model] = []
                model_metrics[model].append(metrics[metric])
        
        if not model_metrics:
            return f"No data found for metric: {metric}"
        
        result = f"Model comparison for '{metric}' in '{experiment_name}':\n\n"
        
        # Calculate statistics
        for model, values in sorted(model_metrics.items()):
            avg_value = sum(values) / len(values)
            max_value = max(values)
            min_value = min(values)
            
            result += f"• {model}\n"
            result += f"  Runs: {len(values)}\n"
            result += f"  Avg: {avg_value:.4f}\n"
            result += f"  Best: {max_value:.4f}\n"
            result += f"  Worst: {min_value:.4f}\n\n"
        
        return result
        
    except Exception as e:
        return f"Error comparing models: {str(e)}"

@mcp.tool()
def log_prompt_version(
    prompt_name: str,
    version: str,
    prompt_text: str,
    metadata: Optional[Dict] = None
) -> str:
    """Log a prompt version for tracking.
    
    Args:
        prompt_name: Name/identifier for the prompt
        version: Version string (e.g., "v1.2")
        prompt_text: The actual prompt text
        metadata: Additional metadata
    """
    try:
        prompt_data = {
            "name": prompt_name,
            "version": version,
            "text": prompt_text,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        response = requests.post(
            f"{OPIK_API_URL}/prompts",
            json=prompt_data,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        prompt_id = response.json().get("id", "unknown")
        
        return f"Logged prompt version: {prompt_name} {version} (ID: {prompt_id})"
        
    except Exception as e:
        return f"Error logging prompt version: {str(e)}"

if __name__ == "__main__":
    import asyncio
    print(f"Opik trace server starting (base URL: {OPIK_BASE_URL})")
    mcp.run()