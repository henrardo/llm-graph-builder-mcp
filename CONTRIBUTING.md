# Contributing to LLM Graph Builder MCP

Thank you for your interest in contributing!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/henrardo/llm-graph-builder-mcp.git
cd llm-graph-builder-mcp
```

2. Install in development mode:
```bash
uv pip install -e .
```

3. Set up the backend (see README.md for full instructions)

## Guidelines

### Core Principle: Zero Backend Modifications

This project's main goal is to provide a clean MCP wrapper that works with the **unmodified** Neo4j LLM Graph Builder backend. 

**When contributing:**
- Avoid changes to `llm-graph-builder/` backend code
- All compatibility logic should live in the MCP server (`llm_graph_builder_mcp/server.py`)
- If backend changes are absolutely necessary, document why and open an issue first

### Code Style

- Use type hints
- Add docstrings to public functions
- Follow PEP 8
- Keep functions focused and readable

### Testing

Before submitting a PR:
1. Test with the included backend version
2. Verify the MCP works with Claude Desktop
3. Test at least one URL type (Wikipedia, PDF, web page, or YouTube)
4. Ensure no regressions in existing functionality

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test thoroughly
5. Commit with clear messages
6. Push to your fork
7. Open a pull request

**PR Description should include:**
- What problem does this solve?
- How was it tested?
- Any breaking changes?
- Screenshots/examples (if applicable)

## Issue Reporting

When opening an issue, please include:
- MCP version
- Backend version (commit hash if not using included version)
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs from:
  - `~/Library/Logs/Claude/mcp-server-llm-graph-builder.log` (macOS)
  - Backend terminal output

## Feature Requests

Open an issue with:
- Clear description of the feature
- Use case / why it's needed
- Proposed implementation (optional)

## Questions?

Open a [GitHub Discussion](https://github.com/henrardo/llm-graph-builder-mcp/discussions) for general questions or join the conversation!

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

