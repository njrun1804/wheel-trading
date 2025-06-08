# Makefile for autonomous development

.PHONY: help install test lint format clean deploy fix push setup

# Default target
help:
	@echo "🤖 Claude Code Autonomous Development Commands:"
	@echo "  make setup    - Initial project setup"
	@echo "  make install  - Install all dependencies"
	@echo "  make test     - Run all tests"
	@echo "  make lint     - Run linters"
	@echo "  make format   - Auto-format code"
	@echo "  make fix      - Fix common issues and commit"
	@echo "  make push     - Push changes (auto-creates PR if needed)"
	@echo "  make deploy   - Deploy to Google Cloud"
	@echo "  make clean    - Clean up generated files"

# Initial setup
setup:
	@echo "Setting up development environment..."
	@which python3 || (echo "Python not found" && exit 1)
	@which node || (echo "Node.js not found" && exit 1)
	@python3 -m venv venv || true
	@source venv/bin/activate || true
	@pip install --upgrade pip || true
	@[ -f requirements.txt ] && pip install -r requirements.txt || true
	@[ -f requirements-dev.txt ] && pip install -r requirements-dev.txt || true
	@[ -f package.json ] && npm install || true
	@echo "✅ Setup complete!"

# Install dependencies
install:
	@[ -f requirements.txt ] && pip install -r requirements.txt || true
	@[ -f package.json ] && npm install || true

# Run tests
test:
	@echo "Running tests..."
	@[ -f package.json ] && npm test --if-present || true
	@[ -d tests ] && python -m pytest tests/ -v || echo "No Python tests found"

# Lint code
lint:
	@echo "Linting code..."
	@[ -f package.json ] && npm run lint --if-present || true
	@find . -name "*.py" -not -path "./venv/*" -not -path "./.git/*" | xargs flake8 --max-line-length=100 || true

# Format code
format:
	@echo "Formatting code..."
	@[ -f package.json ] && npm run format --if-present || true
	@find . -name "*.py" -not -path "./venv/*" -not -path "./.git/*" | xargs black || true
	@find . -name "*.json" -o -name "*.yaml" -o -name "*.yml" | grep -v node_modules | xargs prettier --write || true

# Fix issues and commit
fix: format lint
	@./scripts/dev.sh fix
	@if [ -n "$$(git status --porcelain)" ]; then \
		git add -A && \
		git commit -m "Auto-fix: Code formatting and linting" && \
		echo "✅ Changes committed"; \
	else \
		echo "✅ No changes needed"; \
	fi

# Push changes
push:
	@if git push origin main 2>/dev/null; then \
		echo "✅ Pushed to main"; \
	else \
		BRANCH="auto-$$(date +%Y%m%d-%H%M%S)" && \
		git checkout -b "$$BRANCH" && \
		git push -u origin "$$BRANCH" && \
		gh pr create --fill --base main --head "$$BRANCH" && \
		echo "✅ Created PR from $$BRANCH"; \
	fi

# Deploy to Google Cloud
deploy:
	@echo "Deploying to Google Cloud..."
	@gcloud builds submit --config cloudbuild.yaml || echo "Add application code before deploying"

# Clean up
clean:
	@echo "Cleaning up..."
	@rm -rf node_modules venv __pycache__ .pytest_cache dist build *.egg-info
	@find . -name "*.pyc" -delete
	@find . -name ".DS_Store" -delete
	@echo "✅ Cleaned!"