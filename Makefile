# SPDX-License-Identifier: MIT
# spartan-torch Makefile

.PHONY: help install check lint format typecheck security test test-unit build clean new-block info update-deps

# Default target
help:
	@echo "spartan-torch - Efficient Deep Learning for 4GB VRAM"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Development:"
	@echo "  install      - Install all dependencies with uv"
	@echo "  check        - Quick check (lint + typecheck + unit tests)"
	@echo "  lint         - Run ruff lint"
	@echo "  format       - Run ruff format + fix"
	@echo "  typecheck    - Run ty type checker"
	@echo "  security     - Run bandit security scan"
	@echo ""
	@echo "Testing:"
	@echo "  test         - Run all unit tests"
	@echo "  test-unit    - Run unit tests (stop on first failure)"
	@echo ""
	@echo "Build:"
	@echo "  build        - Build wheel + sdist"
	@echo ""
	@echo "Utility:"
	@echo "  clean        - Clean build artifacts and caches"
	@echo "  new-block    - Scaffold new block (usage: make new-block CATEGORY=attention NAME=MyAttention)"
	@echo "  info         - Show environment info"
	@echo "  update-deps  - Update uv lockfile"

# Installation
install:
	uv sync --extra dev --extra bench

# Quick check (ruff + ty + unit tests)
check: lint typecheck test-unit

# Linting
lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

# Type checking
typecheck:
	uv run ty check src/spartan_torch

# Security
security:
	uv run bandit -r src/spartan_torch

# Testing
test:
	uv run pytest tests/unit -v --tb=short

test-unit:
	uv run pytest tests/unit -x --tb=short

# Build
build:
	uv build --wheel --sdist

# Clean
ifeq ($(OS),Windows_NT)
clean:
	@pwsh -NoProfile -Command "Remove-Item -Recurse -Force -ErrorAction SilentlyContinue 'dist','build','*.egg-info','.pytest_cache','.ruff_cache','.coverage','htmlcov','coverage.xml','.triton_cache','.uv_cache','.venv'; Get-ChildItem -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue; Get-ChildItem -Recurse -Filter '*.pyc' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue"
else
clean:
	rm -rf dist build *.egg-info
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov coverage.xml
	rm -rf .triton_cache .uv_cache .venv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
endif



# Scaffold new block
new-block:
	@if [ -z "$(CATEGORY)" ] || [ -z "$(NAME)" ]; then \
		echo "Usage: make new-block CATEGORY=attention NAME=MyAttention"; \
		echo "Categories: attention, norm, conv, ssm, moe, generative, rl, robotics"; \
		exit 1; \
	fi
	uv run python .opencode/skill/spartan-torch/scripts/new_block.py --category $(CATEGORY) --name $(NAME)

# Update dependencies
update-deps:
	uv lock --upgrade
	uv sync --extra dev --extra bench

# Show environment info
info:
	@echo "Python: $$(python --version)"
	@echo "UV: $$(uv --version)"
	@echo "PyTorch: $$(python -c 'import torch; print(torch.__version__)')"
	@echo "CUDA: $$(python -c 'import torch; print(torch.version.cuda)')"
	@echo "Triton: $$(python -c 'import triton; print(triton.__version__)' 2>/dev/null || echo 'not installed')"
	@echo "FlashAttn: $$(python -c 'import flash_attn; print(flash_attn.__version__)' 2>/dev/null || echo 'not installed')"
