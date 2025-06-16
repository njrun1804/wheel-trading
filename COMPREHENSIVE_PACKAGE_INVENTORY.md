# Comprehensive Python Package Inventory
**Wheel Trading System - Complete Package Analysis**

Generated: 2025-06-16T15:39:57.584427

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Packages** | 412 packages |
| **Python Version** | 3.11.10 |
| **Python Executable** | `/Users/mikeedwards/.pyenv/versions/3.11.10/bin/python3.11` |
| **Total Disk Usage** | 4.3GB (PyEnv: 3.3GB, Local: 996MB) |
| **Editable Packages** | 3 development installations |
| **Key Dependencies** | 16 core packages identified |

## Installation Sources

| Source | Package Count | Disk Usage | Description |
|--------|---------------|------------|-------------|
| **PyEnv** | ~330 packages | 3.3GB | Main Python 3.11.10 environment |
| **Local User** | ~82 packages | 996MB | User-specific installations (.local) |
| **Homebrew** | 8 packages | 3.7MB | System Python 3.13.4 packages |

## Editable/Development Packages

These are packages installed in development mode from local source code:

| Package | Version | Location |
|---------|---------|----------|
| **mcp-py-repl** | 0.1.4 | `/Users/mikeedwards/mcp-servers/community/mcp-py-repl` |
| **mcp-server-scikit-learn** | 0.1.0 | `/Users/mikeedwards/mcp-servers/community/mcp-server-scikit-learn` |
| **mcp-server-stats** | 0.2.2 | `/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/mcp-servers/statsource` |

## Key Package Dependencies

Core packages with their dependency relationships:

| Package | Version | Install Method | Dependencies | Used By | Disk Usage |
|---------|---------|----------------|--------------|---------|------------|
| **numpy** | 1.26.4 | pyenv | 0 | 41 packages | ~50MB |
| **pandas** | 2.3.0 | pyenv | 4 | 15 packages | 79MB |
| **torch** | 2.7.1 | pyenv | 6 | 2 packages | 350MB |
| **tensorflow** | 2.16.2 | pyenv | 22 | 1 package | 1.0GB |
| **mlx** | 0.26.1 | pyenv | 0 | 1 package | 109MB |
| **pydantic** | 2.11.5 | pyenv | 4 | 16 packages | ~30MB |
| **scipy** | 1.13.1 | pyenv | 1 | 14 packages | 111MB |
| **matplotlib** | 3.10.3 | pyenv | 9 | 3 packages | ~40MB |
| **scikit-learn** | 1.7.0 | pyenv | 4 | 5 packages | ~35MB |
| **anthropic** | 0.54.0 | pyenv | 7 | 0 packages | ~15MB |
| **databento** | 0.57.0 | pyenv | 7 | 0 packages | ~25MB |
| **duckdb** | 1.3.0 | pyenv | 0 | 1 package | ~35MB |
| **fastapi** | 0.115.12 | local | 3 | 2 packages | ~20MB |
| **langchain** | 0.3.25 | local | 7 | 0 packages | ~25MB |
| **pytest** | 8.4.0 | pyenv | 4 | 7 packages | ~15MB |
| **uvicorn** | 0.34.3 | local | 2 | 3 packages | ~10MB |

## Package Categories

### AI/ML Frameworks (8 packages)
- **torch** (2.7.1) - 350MB
- **tensorflow** (2.16.2) - 1.0GB  
- **jax** (0.6.1) - part of 231MB jaxlib
- **jaxlib** (0.6.1) - 231MB
- **mlx** (0.26.1) - 109MB
- **transformers** (4.52.4) - 91MB
- **sentence-transformers** (4.1.0)
- **scikit-learn** (1.7.0) - ~35MB

### Data Science (8 packages)
- **numpy** (1.26.4) - Foundation for all numeric computing
- **pandas** (2.3.0) - 79MB, used by 15 other packages
- **scipy** (1.13.1) - 111MB, used by 14 packages
- **matplotlib** (3.10.3) - Plotting and visualization
- **seaborn** (0.13.2) - Statistical visualization
- **plotly** (6.1.2) - Interactive plotting
- **polars** (1.30.0) - Fast DataFrame library
- **pyarrow** (20.0.0) - 111MB, columnar data

### Trading/Finance (6 packages)
- **databento** (0.57.0) - Market data provider
- **alpaca-py** (0.40.1) - Trading API
- **fredapi** (0.5.2) - Federal Reserve Economic Data
- **yfinance** (0.2.62) - Yahoo Finance data
- **QuantLib** (1.38) - Quantitative finance library
- **exchange-calendars** (4.10.1) - Trading calendar utilities

### Web/API Frameworks (8 packages)
- **fastapi** (0.115.12) - High-performance web framework
- **uvicorn** (0.34.3) - ASGI server
- **aiohttp** (3.12.12) - Async HTTP client/server
- **httpx** (0.28.1) - HTTP client
- **Flask** (3.0.3) - Web framework
- **starlette** (0.46.2) - ASGI framework
- **gunicorn** (23.0.0) - WSGI server
- **requests** (2.32.4) - HTTP library

### LLM/AI Services (5 packages)
- **anthropic** (0.54.0) - Anthropic Claude API
- **openai** (1.86.0) - OpenAI API
- **langchain** (0.3.25) - LLM application framework
- **langchain-core** (0.3.65) - Core LangChain components
- **langsmith** (0.3.45) - LangChain monitoring

### Development Tools (7 packages)
- **pytest** (8.4.0) - Testing framework
- **black** (25.1.0) - Code formatter
- **mypy** (1.16.1) - Type checker
- **ruff** (0.11.13) - Fast linter
- **isort** (6.0.1) - Import sorter
- **pre-commit** (4.2.0) - Git hooks
- **coverage** (7.8.2) - Code coverage

### Database (6 packages)
- **duckdb** (1.3.0) - In-memory analytical database
- **SQLAlchemy** (2.0.41) - SQL toolkit and ORM
- **alembic** (1.16.1) - Database migration tool
- **aiosqlite** (0.21.0) - Async SQLite interface
- **asyncpg** (0.30.0) - Async PostgreSQL driver
- **peewee** (3.18.1) - Lightweight ORM

### Observability (4 packages)
- **arize-phoenix** (10.11.0) - ML observability
- **logfire** (3.18.0) - Structured logging
- **mlflow** (3.1.0) - ML lifecycle management
- **opentelemetry-api** (1.34.1) - Observability framework

### MCP Servers (5 packages)
- **mcp** (1.9.4) - Model Context Protocol
- **mcp-py-repl** (0.1.4) - Python REPL server (editable)
- **mcp-server-duckdb** (1.1.0) - DuckDB MCP server
- **mcp-server-scikit-learn** (0.1.0) - ML MCP server (editable)
- **mcp-server-stats** (0.2.2) - Statistics MCP server (editable)

### System/Utils (7 packages)
- **pydantic** (2.11.5) - Data validation (used by 16 packages)
- **click** (8.2.1) - CLI creation toolkit
- **typer** (0.16.0) - Modern CLI framework
- **rich** (14.0.0) - Rich text and beautiful formatting
- **tqdm** (4.67.1) - Progress bars
- **psutil** (7.0.0) - System and process utilities
- **watchdog** (6.0.0) - File system monitoring

## Dependency Analysis

### Most Required Packages (High Impact)
1. **numpy** - Required by 41 packages (foundation of data science stack)
2. **pydantic** - Required by 16 packages (data validation everywhere)
3. **pandas** - Required by 15 packages (data manipulation)
4. **scipy** - Required by 14 packages (scientific computing)
5. **pytest** - Required by 7 packages (testing framework)

### Most Complex Dependencies (High Maintenance)
1. **tensorflow** - Requires 22 packages (heavyweight ML framework)
2. **matplotlib** - Requires 9 packages (complex plotting dependencies)
3. **anthropic** - Requires 7 packages (API client dependencies)
4. **databento** - Requires 7 packages (market data client)
5. **langchain** - Requires 7 packages (LLM framework)

### Largest Packages by Disk Usage
1. **tensorflow** - 1.0GB (complete ML framework with GPU support)
2. **torch** - 350MB (PyTorch deep learning framework)
3. **jaxlib** - 231MB (JAX linear algebra library)
4. **jax_plugins** - 152MB (JAX GPU/TPU plugins)
5. **scipy** - 111MB (scientific computing library)
6. **pyarrow** - 111MB (columnar data format)
7. **mlx** - 109MB (Apple Silicon ML framework)
8. **transformers** - 91MB (Hugging Face transformers)
9. **pandas** - 79MB (data manipulation library)

## Installation Method Breakdown

### PyEnv Environment (Primary)
- **Location**: `/Users/mikeedwards/.pyenv/versions/3.11.10/lib/python3.11/site-packages`
- **Size**: 3.3GB
- **Package Count**: ~330 packages
- **Purpose**: Main development environment with all data science, ML, and trading packages

### Local User Installation
- **Location**: `/Users/mikeedwards/.local/lib/python3.11/site-packages`
- **Size**: 996MB  
- **Package Count**: ~82 packages
- **Purpose**: User-specific packages including web frameworks and utilities

### System/Homebrew Installation
- **Location**: `/opt/homebrew/lib/python3.13/site-packages`
- **Size**: 3.7MB
- **Package Count**: 8 packages
- **Purpose**: Minimal system Python 3.13 with basic packages

## Risk Assessment & Recommendations

### Strengths
- ✅ Well-organized package structure with clear separation
- ✅ Key packages are up-to-date versions
- ✅ Strong foundation packages (numpy, pandas, scipy)
- ✅ Comprehensive ML/AI toolkit
- ✅ Good development tooling (testing, linting, formatting)

### Potential Issues
- ⚠️ Large disk usage (4.3GB total)
- ⚠️ Heavy dependencies (tensorflow: 22 deps, matplotlib: 9 deps)
- ⚠️ Multiple Python environments could cause conflicts
- ⚠️ Some packages have high dependency counts

### Recommendations
1. **Dependency Management**: Consider using dependency pinning for production
2. **Environment Isolation**: Continue using pyenv for environment management
3. **Package Cleanup**: Review unused packages periodically
4. **Monitoring**: Track package updates and security vulnerabilities
5. **Documentation**: Maintain requirements.txt files for reproducibility

## Summary

This is a well-maintained, production-ready Python environment for wheel trading with:
- Complete data science and ML stack
- Professional trading and finance libraries  
- Robust web APIs and database connectivity
- Comprehensive development and testing tools
- Advanced observability and monitoring capabilities

The 412 packages represent a mature, feature-complete environment suitable for advanced quantitative trading operations with AI/ML capabilities.