# 🔧 MCP Text Analyzer Tool — Learn MCP by Building

A hands-on learning project that teaches the **Model Context Protocol (MCP)** by
building a real, working **Text Analyzer** MCP server from scratch.

---

## 📖 What is MCP?

**Model Context Protocol (MCP)** is an open standard (created by Anthropic) that
lets AI assistants (like Claude, Gemini, GitHub Copilot, etc.) connect to external
tools and data sources through a universal protocol.

Think of it like **USB for AI** — just as USB lets any device plug into any computer,
MCP lets any AI model plug into any tool.

```
┌─────────────┐      MCP Protocol       ┌─────────────────┐
│  AI Client   │ ◄──────────────────────► │   MCP Server     │
│  (Claude,    │   JSON-RPC over stdio   │  (Your Tool!)    │
│   Gemini,    │   or HTTP/SSE           │                  │
│   Copilot)   │                         │  - Tools         │
│              │                         │  - Resources     │
└─────────────┘                         │  - Prompts       │
                                         └─────────────────┘
```

### The Three Primitives of MCP

| Primitive     | What it does                              | Analogy                  |
|:-------------|:------------------------------------------|:-------------------------|
| **Tools**     | Functions the AI can call                | API endpoints            |
| **Resources** | Data the AI can read                     | Files / GET endpoints    |
| **Prompts**   | Pre-built prompt templates               | Reusable instruction sets|

This project focuses on **Tools** — the most common and useful primitive.

---

## 🏗️ Project Structure

```
MCP_Text_Analyzer_Tool/
├── README.md                  ← You are here
├── requirements.txt           ← Python dependencies
├── server.py                  ← The MCP server (main file)
├── test_server.py             ← Test script to verify it works
└── mcp_config_example.json    ← Example config for AI clients
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd Fundamentals_Miscellaneous_Learning/MCP_Text_Analyzer_Tool
pip install -r requirements.txt
```

### 2. Run the server (standalone test)

```bash
python test_server.py
```

### 3. Connect to an AI client

Add to your client's MCP config (e.g. Claude Desktop, Antigravity):
```json
{
  "mcpServers": {
    "text-analyzer": {
      "command": "python",
      "args": ["server.py"],
      "cwd": "/path/to/MCP_Text_Analyzer_Tool"
    }
  }
}
```

---

## 🔬 What the Tools Do

This server exposes **3 tools** for the AI to call:

### `analyze_text`
Full text analysis — word count, character count, sentence count, paragraph count,
average word length, estimated reading time, and top 10 most frequent words.

### `count_words`
Simple word counter — returns word count, unique words, and vocabulary richness ratio.

### `readability_score`
Computes the **Flesch Reading Ease** score (0-100) with a human-friendly grade level.

---

## 🧠 Key MCP Concepts Demonstrated

1. **Server creation** — `FastMCP("server-name")` bootstraps everything
2. **Tool registration** — `@mcp.tool()` decorator exposes functions as tools
3. **Type hints** — MCP auto-generates JSON Schema from Python type hints
4. **Docstrings** — The docstring becomes the tool's description for the AI
5. **Transport** — `mcp.run()` starts the stdio transport (default for local tools)

---

## 📚 Further Reading

- [MCP Official Docs](https://modelcontextprotocol.io)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Specification](https://spec.modelcontextprotocol.io)
