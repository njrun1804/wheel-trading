.PHONY: analyze test profile validate clean

# Run position analysis
analyze:
	python run.py -p 100000 --verbose

# Run tests with coverage
test:
	pytest tests/ -v --cov=src --cov-report=html

# Profile performance
profile:
	python -m cProfile -o profile.stats run.py -p 100000
	snakeviz profile.stats

# Memory profiling
memory:
	python -m memory_profiler run.py -p 100000

# Validate data quality
validate:
	python quick_data_assessment.py
	python verify_database_integrity.py

# Check mathematical consistency
math-check:
	python -m pytest tests/test_math.py tests/test_options.py -v

# Clean temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -f profile.stats
	rm -f *.log
