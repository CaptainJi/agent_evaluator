# Agent Evaluator - Project Context

## Project Overview
Agent Evaluator is a comprehensive evaluation framework for AI agents built on top of Ragas. It provides a unified evaluation approach for AI agents, workflows, and RAG applications across multiple platforms including Dify, Bisheng, n8n, Coze, LangChain, and LangGraph. The framework uses configuration-driven evaluation via YML files without requiring code changes.

**Key Features:**
- Configuration-driven evaluation via YML files
- Multi-platform support with unified interface
- Rich evaluation metrics based on Ragas
- Multiple output formats (console, JSON, HTML, CSV)

## Project Architecture
The project follows a modular architecture with the following key directories:
- `src/agent_evaluator/core/` - Core data structures and configuration models
- `src/agent_evaluator/adapters/` - Platform-specific adapters implementing the base interface
- `src/agent_evaluator/evaluator/` - Evaluation engine and metrics registry
- `src/agent_evaluator/loaders/` - Configuration and dataset loading utilities
- `src/agent_evaluator/reporters/` - Output reporters for different formats

## Key Components

### Core Configuration (core/config.py)
Defines Pydantic models for evaluation configuration:
- `APIConfig`: Platform API configuration (key, URL, timeout)
- `EvaluatorLLMConfig`: LLM provider for evaluation metrics
- `OutputConfig`: Output format and save path configuration
- `EvalConfig`: Main evaluation configuration combining all components

### Platform Adapters (adapters/)
Each platform (Dify, Bisheng, etc.) has its own adapter that implements the `PlatformAdapter` abstract base class. The base class defines:
- Async context management for HTTP clients
- Standard `invoke()` method that returns a unified `AdapterResponse` object
- Standardized response format with answer, contexts, tool calls, and metadata

### Configuration Loading (loaders/config_loader.py)
Handles loading of YML configuration files and validates them against the Pydantic models defined in `core/config.py`.

## Building and Running

### Installation
```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Running Evaluations
```bash
# Run an evaluation using a YML configuration file
agent-eval run config.yml
```

### Configuration File Format
Example configuration includes:
- Target platform selection
- API credentials and base URL
- Dataset path
- Evaluation metrics (from Ragas)
- Evaluator LLM configuration
- Output format and path

## Development Conventions

### Adding New Platform Adapters
1. Create a new file in `src/agent_evaluator/adapters/`
2. Inherit from the `PlatformAdapter` base class
3. Implement the `invoke()` method to call the platform's API
4. Return a standardized `AdapterResponse` object

### Testing
The project uses pytest for testing. Tests are organized by component:
- `tests/test_adapters/` - Tests for platform adapters
- `tests/test_evaluator/` - Tests for evaluation logic
- `tests/test_loaders/` - Tests for configuration/data loading

### Dependencies
- `pyproject.toml` is the primary configuration file for dependencies
- `requirements.txt` provides a traditional dependency list
- Core dependencies include Ragas, PyYAML, Pydantic, HTTPX, and Rich

## Command Line Interface
The project provides a CLI entry point defined in `cli.py` as `agent-eval`. The primary command is `run` which takes a configuration file path.

## Output Formats
The system supports multiple output formats through different reporter implementations:
- Console output for immediate feedback
- JSON format for programmatic processing
- HTML format for human-readable reports
- CSV format for data analysis