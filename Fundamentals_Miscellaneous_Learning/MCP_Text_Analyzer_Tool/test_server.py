"""
Test Script for the MCP Text Analyzer
======================================

This script tests the tool functions DIRECTLY (without going through the
MCP protocol), so you can verify the logic works before connecting to an
AI client.

Run it with:
    python test_server.py
"""

import json
import sys

# Import our tool functions directly from server.py
from server import analyze_text, count_words, readability_score


def pretty_print(title: str, result: dict) -> None:
    """Pretty-print a test result."""
    print(f"\n{'='*60}")
    print(f"  🧪 {title}")
    print(f"{'='*60}")
    print(json.dumps(result, indent=2))


def main():
    # ── Sample texts for testing ──
    simple_text = "The cat sat on the mat. The dog chased the cat."

    complex_text = """
    Artificial intelligence has transformed the landscape of modern computing.
    Machine learning algorithms can now process vast quantities of data,
    identifying patterns that would be impossible for humans to detect manually.

    The emergence of large language models has particularly revolutionized
    natural language processing. These models, trained on enormous corpora
    of text data, demonstrate remarkable capabilities in understanding and
    generating human language.

    However, significant challenges remain. Issues of bias, hallucination,
    and energy consumption continue to demand attention from researchers
    and practitioners alike.
    """

    print("\n" + "🔧 MCP Text Analyzer — Test Suite".center(60))
    print("=" * 60)

    # ── Test 1: Full text analysis on simple text ──
    result = analyze_text(simple_text)
    pretty_print("analyze_text (simple)", result)

    # ── Test 2: Full text analysis on complex text ──
    result = analyze_text(complex_text)
    pretty_print("analyze_text (complex)", result)

    # ── Test 3: Word count ──
    result = count_words(complex_text)
    pretty_print("count_words", result)

    # ── Test 4: Readability score ──
    result = readability_score(simple_text)
    pretty_print("readability_score (simple)", result)

    result = readability_score(complex_text)
    pretty_print("readability_score (complex)", result)

    # ── Test 5: Edge case — empty text ──
    result = analyze_text("")
    pretty_print("analyze_text (empty string)", result)

    result = readability_score("")
    pretty_print("readability_score (empty string)", result)

    print(f"\n{'='*60}")
    print("  ✅ All tests completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
