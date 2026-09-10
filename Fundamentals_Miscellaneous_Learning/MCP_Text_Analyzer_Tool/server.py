"""
MCP Text Analyzer Server
=========================

A Model Context Protocol (MCP) server that exposes text analysis tools.
This is a learning project to understand how MCP servers work.

HOW IT WORKS
─────────────
1. We import MCPServer — a high-level wrapper around the MCP protocol.
   (In MCP SDK v1, this was called FastMCP. It was renamed in v2.)
2. We create a server instance with MCPServer("text-analyzer").
3. We register tools using the @mcp.tool() decorator.
4. Each tool is a plain Python function with type hints + a docstring.
   - Type hints → MCP auto-generates the JSON Schema for the tool's inputs
   - Docstring  → becomes the tool description that the AI reads
5. mcp.run() starts the server, listening on stdio (standard in/out).

PROTOCOL FLOW (what happens when an AI calls a tool)
─────────────────────────────────────────────────────
   AI Client                          MCP Server (this file)
       │                                      │
       │──── initialize ─────────────────────►│  (handshake)
       │◄─── server capabilities ─────────────│
       │                                      │
       │──── tools/list ─────────────────────►│  (discover tools)
       │◄─── [analyze_text, count_words, ...] │
       │                                      │
       │──── tools/call("analyze_text",       │  (invoke a tool)
       │      {"text": "Hello world"}) ──────►│
       │◄─── {"word_count": 2, ...} ──────────│
       │                                      │

All communication uses JSON-RPC 2.0 over stdio.
"""

import re
import math
from collections import Counter

# ─── STEP 1: Import MCPServer ─────────────────────────────────────────────────
# MCPServer (formerly FastMCP in v1) is the recommended way to build MCP
# servers in Python. It handles all the protocol details (JSON-RPC,
# capability negotiation, etc.) so you can focus on writing your tool logic.
from mcp.server.mcpserver import MCPServer


# ─── STEP 2: Create the MCP Server ───────────────────────────────────────────
# The name "text-analyzer" identifies this server to AI clients.
# It appears in tool listings and logs.
mcp = MCPServer(
    "text-analyzer",
    # Optional metadata — helps AI clients understand what this server does
    instructions=(
        "A text analysis toolkit. Use 'analyze_text' for comprehensive analysis, "
        "'count_words' for quick word counts, or 'readability_score' for "
        "readability metrics."
    ),
)


# ─── STEP 3: Register Tools ──────────────────────────────────────────────────
# Each @mcp.tool() decorated function becomes a tool that AI clients can call.
# The function name becomes the tool name.
# The docstring becomes the tool description.
# The type hints become the tool's input schema (auto-generated JSON Schema).


@mcp.tool()
def analyze_text(text: str) -> dict:
    """Perform comprehensive text analysis.

    Analyzes the given text and returns detailed statistics including:
    - Word count and character count
    - Sentence and paragraph counts
    - Average word length
    - Estimated reading time
    - Top 10 most frequent words (excluding common stop words)

    Args:
        text: The text content to analyze. Can be any length.

    Returns:
        A dictionary containing all analysis metrics.
    """
    # ── Basic counts ──
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    char_count_no_spaces = len(text.replace(" ", ""))

    # ── Sentence detection (split on . ! ?) ──
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)

    # ── Paragraph detection (split on blank lines) ──
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    paragraph_count = max(len(paragraphs), 1)

    # ── Average word length ──
    avg_word_length = 0.0
    if word_count > 0:
        total_word_chars = sum(len(w.strip(".,!?;:'\"")) for w in words)
        avg_word_length = round(total_word_chars / word_count, 2)

    # ── Reading time (average adult reads ~238 WPM) ──
    reading_time_minutes = round(word_count / 238, 2) if word_count > 0 else 0

    # ── Top words (excluding common English stop words) ──
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "shall", "it", "its", "i",
        "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "our", "their", "this", "that", "these", "those",
        "not", "no", "so", "if", "as", "than", "then",
    }
    cleaned_words = [
        re.sub(r"[^a-zA-Z]", "", w).lower()
        for w in words
    ]
    filtered_words = [w for w in cleaned_words if w and w not in stop_words]
    top_words = Counter(filtered_words).most_common(10)

    return {
        "word_count": word_count,
        "character_count": char_count,
        "character_count_no_spaces": char_count_no_spaces,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "average_word_length": avg_word_length,
        "estimated_reading_time_minutes": reading_time_minutes,
        "top_words": [{"word": w, "count": c} for w, c in top_words],
    }


@mcp.tool()
def count_words(text: str) -> dict:
    """Count words in the given text.

    A lightweight tool that returns basic word count statistics:
    total words, unique words, and vocabulary richness (unique/total ratio).

    Args:
        text: The text to count words in.

    Returns:
        Dictionary with word_count, unique_words, and vocabulary_richness.
    """
    words = text.split()
    word_count = len(words)

    # Normalize words for uniqueness check
    normalized = [re.sub(r"[^a-zA-Z]", "", w).lower() for w in words]
    normalized = [w for w in normalized if w]
    unique_count = len(set(normalized))

    # Vocabulary richness: ratio of unique words to total words
    # Higher = more diverse vocabulary
    richness = round(unique_count / word_count, 4) if word_count > 0 else 0

    return {
        "word_count": word_count,
        "unique_words": unique_count,
        "vocabulary_richness": richness,
    }


@mcp.tool()
def readability_score(text: str) -> dict:
    """Calculate the Flesch Reading Ease score for the given text.

    The Flesch Reading Ease formula measures how easy a text is to read:
    - 90-100: Very easy (5th grade)
    - 80-89:  Easy (6th grade)
    - 70-79:  Fairly easy (7th grade)
    - 60-69:  Standard (8th-9th grade)
    - 50-59:  Fairly difficult (10th-12th grade)
    - 30-49:  Difficult (college level)
    - 0-29:   Very difficult (graduate level)

    Formula: 206.835 - 1.015 × (words/sentences) - 84.6 × (syllables/words)

    Args:
        text: The text to score for readability.

    Returns:
        Dictionary with the score, grade level, and component metrics.
    """
    words = text.split()
    word_count = len(words)

    if word_count == 0:
        return {
            "flesch_reading_ease": 0,
            "grade_level": "N/A",
            "interpretation": "No text provided",
            "word_count": 0,
            "sentence_count": 0,
            "syllable_count": 0,
        }

    # Count sentences
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = max(len(sentences), 1)

    # Count syllables (approximate heuristic)
    def count_syllables(word: str) -> int:
        word = word.lower().strip(".,!?;:'\"")
        if not word:
            return 0
        # Remove trailing 'e' (silent e)
        if word.endswith("e") and len(word) > 2:
            word = word[:-1]
        # Count vowel groups
        vowel_groups = re.findall(r"[aeiouy]+", word)
        count = len(vowel_groups)
        return max(count, 1)

    total_syllables = sum(count_syllables(w) for w in words)

    # Flesch Reading Ease formula
    score = (
        206.835
        - 1.015 * (word_count / sentence_count)
        - 84.6 * (total_syllables / word_count)
    )
    score = round(max(0, min(100, score)), 2)  # Clamp to 0-100

    # Map score to grade level
    if score >= 90:
        grade = "5th grade"
        interpretation = "Very easy to read"
    elif score >= 80:
        grade = "6th grade"
        interpretation = "Easy to read"
    elif score >= 70:
        grade = "7th grade"
        interpretation = "Fairly easy to read"
    elif score >= 60:
        grade = "8th-9th grade"
        interpretation = "Standard / plain English"
    elif score >= 50:
        grade = "10th-12th grade"
        interpretation = "Fairly difficult to read"
    elif score >= 30:
        grade = "College level"
        interpretation = "Difficult to read"
    else:
        grade = "Graduate level"
        interpretation = "Very difficult to read"

    return {
        "flesch_reading_ease": score,
        "grade_level": grade,
        "interpretation": interpretation,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "syllable_count": total_syllables,
        "avg_words_per_sentence": round(word_count / sentence_count, 2),
        "avg_syllables_per_word": round(total_syllables / word_count, 2),
    }


# ─── STEP 4: Run the Server ──────────────────────────────────────────────────
# mcp.run() starts the server and listens for connections.
# By default it uses "stdio" transport — the AI client spawns this as a
# subprocess and communicates via stdin/stdout using JSON-RPC 2.0.
#
# Transport options:
#   stdio            → Local clients (Claude Desktop, Antigravity, etc.)
#   streamable-http  → Remote clients (ChatGPT via ngrok, web apps, etc.)
#
# Usage:
#   python server.py              # stdio mode (default)
#   python server.py --http       # HTTP mode on port 8000
#   python server.py --http 3000  # HTTP mode on custom port
if __name__ == "__main__":
    import sys

    if "--http" in sys.argv:
        # Get optional port argument (default: 8000)
        idx = sys.argv.index("--http")
        port = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 8000
        print(f"🌐 Starting MCP server in HTTP mode on port {port}...")
        print(f"   Endpoint: http://localhost:{port}/mcp")
        print(f"   Use ngrok to expose: ngrok http {port}")
        mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
    else:
        mcp.run()  # Default: stdio transport
