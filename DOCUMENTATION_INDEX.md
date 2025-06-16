# Documentation Index

Unity Wheel Trading Bot v2.2 - Complete documentation guide and navigation.

## 📚 Core Documentation

### Getting Started
- [README.md](README.md) - Project overview and quick start
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Common commands and operations
- [CLAUDE.md](CLAUDE.md) - Claude Code AI assistant instructions

### Setup and Development
- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - External service integration setup
- [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) - Complete development workflow
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [HOUSEKEEPING_GUIDE.md](HOUSEKEEPING_GUIDE.md) - File organization rules

### Architecture and Design
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture overview
- [docs/API_GUIDE.md](docs/API_GUIDE.md) - Python API reference
- [docs/ORCHESTRATOR_GUIDE.md](docs/ORCHESTRATOR_GUIDE.md) - MCP orchestrator documentation

## 🔧 Technical Guides

### API and Integration
- [docs/API_GUIDE.md](docs/API_GUIDE.md) - External API usage patterns
- [docs/MCP_COMPLETE_GUIDE.md](docs/MCP_COMPLETE_GUIDE.md) - Model Context Protocol guide
- [docs/DATABENTO_UNITY_GUIDE.md](docs/DATABENTO_UNITY_GUIDE.md) - Databento integration

### Data Management
- [docs/DATABASE_ARCHITECTURE.md](docs/DATABASE_ARCHITECTURE.md) - Database design
- [docs/LIVE_DATA_SAFEGUARDS.md](docs/LIVE_DATA_SAFEGUARDS.md) - Live data protection
- [docs/OPTIMAL_DATA_STRUCTURE.md](docs/OPTIMAL_DATA_STRUCTURE.md) - Data optimization

### Financial Modeling
- [docs/ADVANCED_FINANCIAL_MODELING_GUIDE.md](docs/ADVANCED_FINANCIAL_MODELING_GUIDE.md) - Advanced models
- [docs/BORROWING_COST_ANALYSIS_GUIDE.md](docs/BORROWING_COST_ANALYSIS_GUIDE.md) - Borrowing analysis

### System Optimization
- [HARDWARE_ACCELERATION_SETUP.md](HARDWARE_ACCELERATION_SETUP.md) - Hardware optimization
- [docs/OPTIMIZATION_IMPLEMENTATION_SUMMARY.md](docs/OPTIMIZATION_IMPLEMENTATION_SUMMARY.md) - Performance optimization

## 🏗️ Architecture Decision Records (ADRs)

- [docs/adr/001-confidence-scores.md](docs/adr/001-confidence-scores.md) - Confidence scoring system
- [docs/adr/002-unified-position-sizing.md](docs/adr/002-unified-position-sizing.md) - Position sizing approach
- [docs/adr/003-no-bare-exceptions.md](docs/adr/003-no-bare-exceptions.md) - Exception handling standards

## 📖 User Guides

### Trading Operations
- [SINGLE_ACCOUNT_README.md](SINGLE_ACCOUNT_README.md) - Single account usage
- [examples/single_account_simple.py](examples/single_account_simple.py) - Simple usage example

### Configuration
- [config.yaml](config.yaml) - Main configuration file
- [examples/core/conservative_config.yaml](examples/core/conservative_config.yaml) - Conservative settings

## 🛠️ Development Tools

### Code Quality
- [docs/DOCUMENTATION_TEMPLATE.md](docs/DOCUMENTATION_TEMPLATE.md) - Documentation template
- [pytest.ini](pytest.ini) - Test configuration
- [pyproject.toml](pyproject.toml) - Project metadata

### Scripts and Utilities
- [scripts/health_check.sh](scripts/health_check.sh) - System health verification
- [scripts/setup-mac-m4-optimizations.sh](scripts/setup-mac-m4-optimizations.sh) - Hardware optimization

## 📝 Examples and Tutorials

### Core Examples
- [examples/core/](examples/core/) - Core functionality examples
- [examples/auth/](examples/auth/) - Authentication examples
- [examples/data/](examples/data/) - Data provider examples

### Integration Examples
- [examples/bolt_integration_demo.py](examples/bolt_integration_demo.py) - Bolt integration
- [examples/memory_trading_integration.py](examples/memory_trading_integration.py) - Memory integration

## 🔍 Troubleshooting and Support

### Common Issues
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common problems and solutions (if exists)
- [docs/async_sync_boundaries.md](docs/async_sync_boundaries.md) - Async/sync patterns

### Validation and Testing
- [VALIDATION_TOOLS_README.md](VALIDATION_TOOLS_README.md) - Validation tools
- [tests/](tests/) - Test suite directory

## 📊 Performance and Monitoring

### Performance Analysis
- [PERFORMANCE_ANALYSIS_README.md](PERFORMANCE_ANALYSIS_README.md) - Performance analysis
- [docs/SLICE_CACHE.md](docs/SLICE_CACHE.md) - Caching system

### Hardware Optimization
- [MAC_ACCELERATION_AUDIT.md](MAC_ACCELERATION_AUDIT.md) - macOS optimization
- [docs/METAL_MLX_BUFFER_ALIGNMENT_GUIDE.md](docs/METAL_MLX_BUFFER_ALIGNMENT_GUIDE.md) - Metal GPU optimization

## 🧪 Advanced Topics

### System Components
- [META_SYSTEM_README.md](META_SYSTEM_README.md) - Meta-programming system
- [META_CAPABILITIES.md](META_CAPABILITIES.md) - Meta system capabilities
- [meta/BUILD_INSTRUCTIONS.md](meta/BUILD_INSTRUCTIONS.md) - Meta system setup

### Bolt System
- [bolt/README.md](bolt/README.md) - Bolt system documentation
- [BOLT_README.md](BOLT_README.md) - Bolt overview
- [BOLT_TROUBLESHOOTING.md](BOLT_TROUBLESHOOTING.md) - Bolt troubleshooting

## 📋 Project Management

### Planning and Roadmaps
- [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md) - Production roadmap
- [WHEEL_OPTIMIZATION_FRAMEWORK.md](WHEEL_OPTIMIZATION_FRAMEWORK.md) - Optimization framework

### Git and Version Control
- [GIT_CHEAT_SHEET.md](GIT_CHEAT_SHEET.md) - Git commands reference
- [LICENSE](LICENSE) - Project license

## 🔗 External Resources

### Data Providers
- [Databento Documentation](https://databento.com/docs)
- [FRED API Documentation](https://fred.stlouisfed.org/docs/api/)

### Development Tools
- [Python Type Hints Guide](https://docs.python.org/3/library/typing.html)
- [Pytest Documentation](https://docs.pytest.org/)

## 📚 Documentation Guidelines

### Writing Standards
1. **Use the template**: [docs/DOCUMENTATION_TEMPLATE.md](docs/DOCUMENTATION_TEMPLATE.md)
2. **Include navigation**: Add "Related Documentation" sections
3. **Version consistently**: Use v2.2 for current version
4. **Link properly**: Use relative paths for internal links
5. **Test examples**: Ensure all code examples work

### Command Format Standards
- Use `bash` code blocks for shell commands
- Use `python` code blocks for Python code
- Use `yaml` code blocks for configuration
- Include comments explaining complex commands
- Use consistent option formatting (e.g., `--option value`)

### File Organization
- Core docs in root directory
- Technical docs in `docs/` directory
- Examples in `examples/` directory
- Scripts in `scripts/` directory

---

**Last Updated**: June 2025  
**Version**: 2.2