#!/usr/bin/env python3
"""Lightweight API server for AI development tools.

Provides HTTP endpoints for autonomous AI developers to interact
with the Unity Wheel Trading Bot development environment.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from uvicorn import run as uvicorn_run
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from src.unity_wheel.utils.logging import StructuredLogger
import logging

logger = StructuredLogger(logging.getLogger(__name__))


if HAS_FASTAPI:
    app = FastAPI(
        title="Unity Wheel AI Development API",
        description="API for autonomous AI developers",
        version="1.0.0"
    )
    
    @app.get("/")
    async def root():
        """API status endpoint."""
        return {
            "service": "Unity Wheel AI Development API",
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "endpoints": [
                "/health",
                "/quality",
                "/tests",
                "/coverage", 
                "/performance",
                "/workflows/pre-commit",
                "/workflows/ci",
                "/workflows/optimization"
            ]
        }
    
    @app.get("/health")
    async def get_system_health():
        """Get comprehensive system health status."""
        try:
            from scripts.ai_dev_assistant import AIDevAssistant
            assistant = AIDevAssistant()
            
            health = assistant.validate_system_health()
            return JSONResponse(content=health)
            
        except Exception as e:
            logger.error("health_endpoint_failed", extra={"error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/quality")
    async def get_code_quality():
        """Get code quality analysis."""
        try:
            from scripts.ai_dev_assistant import AIDevAssistant
            assistant = AIDevAssistant()
            
            quality = assistant.analyze_code_quality()
            return JSONResponse(content=quality)
            
        except Exception as e:
            logger.error("quality_endpoint_failed", extra={"error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/tests")
    async def run_tests(filter: str = None, include_slow: bool = False):
        """Execute test suite."""
        try:
            from scripts.ai_dev_assistant import AIDevAssistant
            assistant = AIDevAssistant()
            
            tests = assistant.run_test_suite(
                filter_pattern=filter,
                include_slow=include_slow
            )
            return JSONResponse(content=tests)
            
        except Exception as e:
            logger.error("tests_endpoint_failed", extra={"error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/coverage")
    async def get_test_coverage():
        """Get test coverage analysis."""
        try:
            from scripts.ai_dev_assistant import AIDevAssistant
            assistant = AIDevAssistant()
            
            coverage = assistant.analyze_test_coverage()
            return JSONResponse(content=coverage)
            
        except Exception as e:
            logger.error("coverage_endpoint_failed", extra={"error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/performance")
    async def get_performance_profile():
        """Get performance profiling data."""
        try:
            from scripts.ai_dev_assistant import AIDevAssistant
            assistant = AIDevAssistant()
            
            performance = assistant.generate_performance_profile()
            return JSONResponse(content=performance)
            
        except Exception as e:
            logger.error("performance_endpoint_failed", extra={"error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/workflows/pre-commit")
    async def execute_precommit_workflow():
        """Execute pre-commit workflow."""
        try:
            from scripts.autonomous_workflow import AutonomousWorkflow
            workflow = AutonomousWorkflow()
            
            result = await workflow.pre_commit_workflow()
            return JSONResponse(content=result)
            
        except Exception as e:
            logger.error("precommit_workflow_failed", extra={"error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/workflows/ci")
    async def execute_ci_workflow():
        """Execute continuous integration workflow."""
        try:
            from scripts.autonomous_workflow import AutonomousWorkflow
            workflow = AutonomousWorkflow()
            
            result = await workflow.continuous_integration_workflow()
            return JSONResponse(content=result)
            
        except Exception as e:
            logger.error("ci_workflow_failed", extra={"error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/workflows/optimization")
    async def execute_optimization_workflow():
        """Execute optimization workflow."""
        try:
            from scripts.autonomous_workflow import AutonomousWorkflow
            workflow = AutonomousWorkflow()
            
            result = await workflow.optimization_workflow()
            return JSONResponse(content=result)
            
        except Exception as e:
            logger.error("optimization_workflow_failed", extra={"error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/fixes/auto")
    async def execute_automated_fixes():
        """Execute all available automated fixes."""
        try:
            from scripts.ai_dev_assistant import AIDevAssistant
            assistant = AIDevAssistant()
            
            fixes = assistant.execute_automated_fixes()
            return JSONResponse(content=fixes)
            
        except Exception as e:
            logger.error("auto_fixes_failed", extra={"error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/state/export")
    async def export_system_state():
        """Export complete system state for analysis."""
        try:
            from scripts.ai_dev_assistant import AIDevAssistant
            assistant = AIDevAssistant()
            
            state = assistant.export_system_state()
            return JSONResponse(content=state)
            
        except Exception as e:
            logger.error("state_export_failed", extra={"error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))


class SimpleAPIServer:
    """Fallback simple HTTP server when FastAPI is not available."""
    
    def __init__(self, port: int = 8000):
        self.port = port
        
    async def start(self):
        """Start simple HTTP server."""
        print(f"⚠️  FastAPI not available, starting simple server on port {self.port}")
        print("   Install FastAPI for full API functionality: pip install fastapi uvicorn")
        
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import threading
            
            class APIHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == '/':
                        response = {
                            "service": "Unity Wheel AI Development API (Simple)",
                            "status": "limited",
                            "message": "Install FastAPI for full functionality"
                        }
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(response).encode())
                    else:
                        self.send_response(404)
                        self.end_headers()
            
            server = HTTPServer(('localhost', self.port), APIHandler)
            print(f"🚀 Simple API server running at http://localhost:{self.port}")
            
            # Run in thread to not block
            def run_server():
                server.serve_forever()
            
            thread = threading.Thread(target=run_server, daemon=True)
            thread.start()
            
            # Keep alive
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            print(f"❌ Failed to start simple server: {e}")


async def main():
    """Main entry point for AI API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unity Wheel AI Development API Server")
    parser.add_argument('--port', '-p', type=int, default=8000, help='Port to run server on')
    parser.add_argument('--host', type=str, default='localhost', help='Host to bind to')
    
    args = parser.parse_args()
    
    if HAS_FASTAPI:
        print(f"🚀 Starting Unity Wheel AI Development API server...")
        print(f"📍 Server will be available at: http://{args.host}:{args.port}")
        print(f"📚 API documentation: http://{args.host}:{args.port}/docs")
        print()
        
        logger.info("api_server_starting", extra={
            "host": args.host,
            "port": args.port
        })
        
        # Start FastAPI server
        uvicorn_run(
            "scripts.ai_api_server:app",
            host=args.host,
            port=args.port,
            reload=False,
            log_level="info"
        )
    else:
        # Fallback to simple server
        simple_server = SimpleAPIServer(args.port)
        await simple_server.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 API server stopped.")
    except Exception as e:
        print(f"\n❌ Server error: {e}")