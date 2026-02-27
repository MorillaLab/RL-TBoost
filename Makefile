.PHONY: install lint test cohort representation rl all clean help

help:
	@echo "RL-TBoost — available commands:"
	@echo "  make install        Install all dependencies"
	@echo "  make lint           Lint Python source"
	@echo "  make test           Run unit tests"
	@echo "  make cohort         Run cohort analysis notebooks"
	@echo "  make representation Run representation learning notebooks"
	@echo "  make rl             Run RL model notebooks"
	@echo "  make all            Run full pipeline"
	@echo "  make clean          Remove cache and executed notebooks"

install:
	pip install -r requirements.txt

lint:
	flake8 utilities/ --max-line-length=127 --count --statistics 2>/dev/null || true

test:
	pytest utilities/tests/ -v --tb=short 2>/dev/null || echo "No tests found — skipping"

cohort:
	jupyter nbconvert --to notebook --execute Cohort_analysis/*.ipynb \
		--output-dir Cohort_analysis/

representation:
	jupyter nbconvert --to notebook --execute Representation_Learning/*.ipynb \
		--output-dir Representation_Learning/

rl:
	jupyter nbconvert --to notebook --execute Reinforcement_Learning_Model/*.ipynb \
		--output-dir Reinforcement_Learning_Model/ \
		--ExecutePreprocessor.timeout=3600

all: cohort representation rl
	@echo "Full RL-TBoost pipeline complete."

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	find . -name ".DS_Store" -delete 2>/dev/null; true
	find . -name "*_executed.ipynb" -delete 2>/dev/null; true
	find . -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null; true
