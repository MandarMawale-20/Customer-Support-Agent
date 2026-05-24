# Use a POSIX shell for portable recipes.
SHELL := /bin/sh

# Compose command can be overridden if your environment uses docker-compose.
COMPOSE ?= docker compose
ROOT_DIR := $(CURDIR)

# Default shell command for the local venv on each platform.
ifeq ($(OS),Windows_NT)
WIN_ROOT_DIR := $(subst /,\,$(ROOT_DIR))
LOCAL_CAG_TEST = cmd /c "cd /d \"$(WIN_ROOT_DIR)\CAG\" && call ..\venv\Scripts\activate && pytest tests/ -v --tb=short"
LOCAL_RAG_TEST = cmd /c "cd /d \"$(WIN_ROOT_DIR)\RAG\" && call ..\venv\Scripts\activate && pytest tests/ -v --tb=short"
else
LOCAL_CAG_TEST = cd "$(ROOT_DIR)/CAG" && . ../venv/bin/activate && pytest tests/ -v --tb=short
LOCAL_RAG_TEST = cd "$(ROOT_DIR)/RAG" && . ../venv/bin/activate && pytest tests/ -v --tb=short
endif

# Run the CAG API and its order service together.
run-cag:
	$(COMPOSE) --profile cag up --build cag-agent cag-orders

# Run the RAG API and its order service together.
run-rag:
	$(COMPOSE) --profile rag up --build rag-agent rag-orders

# Build the shared CAG and RAG images.
build:
	$(COMPOSE) build cag-agent rag-agent

# Run the full demo in one container.
run:
	$(COMPOSE) up --build app

# Run only the CAG test suite through Docker.
test-cag:
	$(COMPOSE) run --rm cag-test

# Run only the RAG test suite through Docker.
test-rag:
	$(COMPOSE) run --rm rag-test

# Run both Docker test suites in sequence and stop on the first failure.
test:
	@status=0; \
	echo "Running CAG tests..."; \
	$(COMPOSE) run --rm cag-test || status=1; \
	if [ $$status -eq 0 ]; then echo "CAG: PASS"; else echo "CAG: FAIL"; fi; \
	echo "Running RAG tests..."; \
	$(COMPOSE) run --rm rag-test || status=1; \
	if [ $$status -eq 0 ]; then echo "RAG: PASS"; else echo "RAG: FAIL"; fi; \
	if [ $$status -eq 0 ]; then echo "Summary: PASS"; else echo "Summary: FAIL"; fi; \
	exit $$status

# Run the CAG tests in the local virtual environment.
local-test-cag:
	$(LOCAL_CAG_TEST)

# Run the RAG tests in the local virtual environment.
local-test-rag:
	$(LOCAL_RAG_TEST)

# Show available Make targets.
help:
	@echo "Available targets:"
	@echo "  run             Start the unified app service with Docker Compose"
	@echo "  run-cag         Start cag-agent and cag-orders with Docker Compose"
	@echo "  run-rag         Start rag-agent and rag-orders with Docker Compose"
	@echo "  build           Build the shared CAG and RAG images"
	@echo "  test            Run CAG tests, then RAG tests, with PASS/FAIL summary"
	@echo "  test-cag        Run only the CAG Docker test suite"
	@echo "  test-rag        Run only the RAG Docker test suite"
	@echo "  local-test-cag   Run CAG tests from the local venv"
	@echo "  local-test-rag   Run RAG tests from the local venv"
	@echo "  help            Print this help text"
