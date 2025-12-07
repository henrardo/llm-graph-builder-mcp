# LLM Graph Builder MCP

Build knowledge graphs from any URL using Claude Desktop and Neo4j.

## What is this?

This [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server enables Claude to automatically extract entities and relationships from unstructured text and build knowledge graphs in Neo4j. Simply give Claude a URL (Wikipedia article, PDF, web page, YouTube video) and ask it to build a knowledge graph - it handles the rest.

**Perfect for:** Research, Zotero integrations, academic papers, content analysis, and building structured knowledge from unstructured sources.

## What's Included

This repository is a **complete, ready-to-use package** containing:
- **llm_graph_builder_mcp/** - The MCP server code
- **llm-graph-builder/** - Neo4j's LLM Graph Builder backend (June 24, 2025, commit `4d7bb5e8`)

Both are included so you get a tested, working version out of the box. Just clone once and you're ready to go!

**Why include the backend?**
- Guaranteed compatibility - this MCP is tested with this exact backend version
- Zero configuration headaches - everything just works together
- If Neo4j updates their backend, you still have a working version

## Features

- **Multi-source support**: Wikipedia, PDFs, web pages, YouTube videos
- **Academic mode**: Extract citations, authors, journals, and bibliographic data
- **Custom schemas**: Define allowed entity types and relationships
- **Community detection**: Find clusters and groups in your knowledge graph
- **Zero setup**: Works with unmodified [llm-graph-builder](https://github.com/neo4j-labs/llm-graph-builder) backend
- **Local processing**: Your data, your Neo4j instance, your control

## Quick Start

### Prerequisites

1. **Neo4j database** - Get a free instance at [Neo4j AuraDB](https://neo4j.com/cloud/aura/)
   - Create an instance and note your connection URI, username, and password
2. **OpenAI API key** - [Get one here](https://platform.openai.com/api-keys)
3. **Python 3.10+** with `uv` - [Install uv](https://docs.astral.sh/uv/)
4. **Claude Desktop** - [Download here](https://claude.ai/download)

### Step 1: Clone This Repository

```bash
# Clone the entire project (includes both MCP and backend)
git clone https://github.com/your-username/llm-graph-builder-mcp.git
cd llm-graph-builder-mcp
```

Your directory structure will be:
```
llm-graph-builder-mcp/           # The MCP server
llm-graph-builder/               # The backend (included)
```

### Step 2: Set Up the Backend

```bash
# Navigate to backend
cd llm-graph-builder/backend

# Create environment file
cp example.env .env
```

**Edit `.env`** with your credentials:
```bash
# Neo4j Connection (from your AuraDB instance)
NEO4J_URI=neo4j+s://your-instance-id.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-auradb-password
NEO4J_DATABASE=neo4j

# OpenAI Configuration
LLM_MODEL_CONFIG_openai_gpt_4.1=gpt-4-turbo-2024-04-09,sk-your-openai-api-key
```

**Install and start the backend:**
```bash
# Create virtual environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Start the backend server
uvicorn score:app --reload --port 8000
```

**Keep this terminal running!** The backend must be running for the MCP to work.

### Step 3: Install the MCP

Open a **new terminal** (keep the backend running in the first one):

```bash
# Navigate to the MCP directory
cd llm-graph-builder-mcp

# Install the MCP
uvx --from . llm-graph-builder-mcp
```

### Step 4: Configure Claude Desktop

Edit your Claude Desktop config file:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Add this configuration:

```json
{
  "mcpServers": {
    "llm-graph-builder": {
      "command": "uvx",
      "args": [
        "--from",
        "/absolute/path/to/llm-graph-builder-mcp",
        "llm-graph-builder-mcp"
      ],
      "env": {
        "NEO4J_URI": "neo4j+s://your-instance-id.databases.neo4j.io",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "your-auradb-password",
        "NEO4J_DATABASE": "neo4j",
        "GRAPH_BUILDER_URL": "http://localhost:8000"
      }
    }
  }
}
```

**⚠️ Important:**
- Replace `/absolute/path/to/` with the full path to your `llm-graph-builder-mcp` directory
  - Run `pwd` in the `llm-graph-builder-mcp` directory to get this path
  - Example: `/Users/yourname/projects/llm-graph-builder-mcp`
- Use the **same credentials** as in your backend `.env` file

### Step 5: Restart Claude Desktop

**Completely quit and restart Claude Desktop** for the changes to take effect.

### Step 6: Test It!

In Claude Desktop, try:
```
Build a knowledge graph from this Wikipedia article:
https://en.wikipedia.org/wiki/The_Hitchhiker%27s_Guide_to_the_Galaxy
```

Claude should now use the MCP to build a knowledge graph in your Neo4j database!

## Usage Examples

### Basic Usage

```
Build a knowledge graph from this Wikipedia article:
https://en.wikipedia.org/wiki/The_Hitchhiker%27s_Guide_to_the_Galaxy
```

### Academic Papers (with citations)

```
Build a knowledge graph from this PDF with bibliographic extraction:
https://example.com/research-paper.pdf
```

### Custom Schema

```
Build a knowledge graph from this article with these entities:
- Nodes: Person, Organization, Location, Event
- Relationships: Person WORKS_FOR Organization, Person ATTENDED Event

https://example.com/article
```

### With Community Detection

```
Build a knowledge graph from this page and enable community detection:
https://en.wikipedia.org/wiki/Renaissance
```

## Querying the Graph

Use the separate [mcp-neo4j-cypher](https://github.com/neo4j/mcp-neo4j) server to query your knowledge graph:

```
# After building a graph, ask Claude:
"What entities are connected to Arthur Dent?"
"Show me all the citations in my research papers"
"Find communities in the knowledge graph"
```

## Tool Reference

### `build_knowledge_graph_from_url`

Extracts entities and relationships from a URL and builds a knowledge graph.

**Parameters:**
- `url` (required): URL to process (Wikipedia, PDF, web page, YouTube)
- `model` (optional): LLM model to use (default: `openai_gpt_4.1`)
- `allowed_nodes` (optional): Comma-separated entity types (e.g., `"Person,Organization,Location"`)
- `allowed_relationships` (optional): Relationship triples (e.g., `"Person,WORKS_FOR,Organization"`)
- `enable_communities` (optional): Enable community detection (default: `false`)
- `extract_bibliographic_info` (optional): Extract academic citations and references (default: `false`)

## Supported Sources

| Type | Example | Notes |
|------|---------|-------|
| Wikipedia | `https://en.wikipedia.org/wiki/...` | Any language supported |
| PDF URLs | `https://example.com/paper.pdf` | Full text extraction |
| Web pages | `https://example.com/article` | Any accessible page |
| YouTube | `https://www.youtube.com/watch?v=...` | Extracts from transcript |

## Architecture

```
Claude Desktop
    ↓ MCP Protocol
llm-graph-builder-mcp (this repo)
    ↓ HTTP
llm-graph-builder backend (FastAPI)
    ↓ Cypher
Neo4j Database
    ↑ Cypher Queries
mcp-neo4j-cypher (separate MCP)
    ↑ MCP Protocol
Claude Desktop
```

## Research & Zotero Integration

This MCP is perfect for academic research workflows:

1. **Export PDF URLs** from your Zotero library
2. **Ask Claude** to process them with bibliographic extraction
3. **Query relationships** between papers, authors, and concepts
4. **Discover connections** in your research

Example:
```
"Build knowledge graphs from these papers with bibliographic extraction:
- https://paper1.pdf
- https://paper2.pdf
- https://paper3.pdf

Then show me how they cite each other and what common themes they share."
```

## 🔄 Backend Version & Updates

This repository includes `llm-graph-builder` from **June 24, 2025** (commit `4d7bb5e8`). This version is tested and fully compatible with the MCP.

### Using the Included Backend (Recommended)

The included backend is frozen at a known-good version. This ensures:
- Everything works out of the box
- No compatibility issues
- Predictable behavior

### Using a Newer Backend Version

If you want to use the latest llm-graph-builder:

```bash
# Remove the included backend
rm -rf llm-graph-builder

# Clone the latest version
git clone https://github.com/neo4j-labs/llm-graph-builder.git

# Follow the same setup steps in Step 2
```

**Note:** Newer versions *should* work (the MCP uses standard endpoints), but haven't been tested. If you encounter issues, revert to the included version.

## Troubleshooting

### Backend won't start
```bash
cd llm-graph-builder/backend
source .venv/bin/activate
uvicorn score:app --reload --port 8000
```

### Claude doesn't see the MCP
1. Check config path is correct (use absolute path, not `~`)
2. Completely quit and restart Claude Desktop (not just close the window)
3. Check Claude logs: `~/Library/Logs/Claude/mcp*.log` (macOS)
4. Verify the MCP path in config matches your actual directory

### "Model not found" error
Make sure your backend `.env` has:
```bash
LLM_MODEL_CONFIG_openai_gpt_4.1=gpt-4-turbo-2024-04-09,YOUR-API-KEY
```

### Backend shows "Connection refused"
- Ensure the backend is running on port 8000
- Check `GRAPH_BUILDER_URL` in Claude config matches the backend URL
- Backend must be running **before** you use the MCP

### Empty graph / few entities
- Enable `extract_bibliographic_info` for academic papers
- Check OpenAI API key is valid and has credits
- Verify Neo4j connection in backend `.env`
- For PDFs: URL must be directly accessible (no authentication required)

## Development

```bash
# Install in development mode
git clone https://github.com/your-username/llm-graph-builder-mcp.git
cd llm-graph-builder-mcp
uv pip install -e .
```

## How It Works

### PDF URLs
The MCP automatically detects PDF URLs, downloads them, and uploads to the backend for full-text extraction using PyMuPDF. No binary garbage, just clean text!

### Academic Extraction
When `extract_bibliographic_info=true`, the MCP instructs the LLM to specifically extract:
- Authors, titles, journals, years, DOIs
- Citations and references
- Research concepts and methods
- Relationships: AUTHORED, CITES, PUBLISHED_IN, DISCUSSES

### Schema Specification
Define allowed entities and relationships to guide extraction:
```
allowed_nodes: "Person,Organization,Product"
allowed_relationships: "Person,FOUNDED,Organization,Organization,PRODUCES,Product"
```

### Zero Backend Modifications
This MCP works with the **unmodified** llm-graph-builder. It uses clever compatibility tricks (like sending a space character for optional parameters) to work seamlessly with the original backend.

## Contributing

Contributions welcome! This project aims to be a clean wrapper with zero backend modifications required.

## Credits

- [Neo4j LLM Graph Builder](https://github.com/neo4j-labs/llm-graph-builder)
- [FastMCP](https://github.com/jlowin/fastmcp)
- [Model Context Protocol](https://modelcontextprotocol.io/)

## Links

- **Issues**: [GitHub Issues](https://github.com/your-username/llm-graph-builder-mcp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/llm-graph-builder-mcp/discussions)
- **Backend**: [llm-graph-builder](https://github.com/neo4j-labs/llm-graph-builder)
- **Query MCP**: [mcp-neo4j-cypher](https://github.com/neo4j/mcp-neo4j)
