#!/usr/bin/env python3
"""Test the word boundary logic from _apply_overlap"""

text = "Instead of trusting retrieved chunks blindly, the LLM evaluates its own inputs in multiple stages: Assesses retrieval quality. Selects key sentences with justification. Synthesizes reasoning paths into final answers."

# Simulate what _apply_overlap does
overlap_tokens = 35
overlap_char_count = min(overlap_tokens * 6, len(text) // 3)

print(f"Text length: {len(text)}")
print(f"Overlap tokens: {overlap_tokens}")
print(f"Overlap char count: {overlap_char_count}")
print()

# Get overlap text from end
overlap_text = text[-overlap_char_count:]
print(f"Raw overlap text ({len(overlap_text)} chars):")
print(repr(overlap_text))
print()

# Find word boundary
boundary_pos = len(overlap_text) - 1
print(f"Starting boundary search from position {boundary_pos}")
while boundary_pos > 0 and overlap_text[boundary_pos] not in (' ', '\n', '\t', '.', '!', '?', ',', ';', ':'):
    boundary_pos -= 1
    
print(f"Found boundary at position {boundary_pos}")
print(f"Character at boundary: {repr(overlap_text[boundary_pos])}")

if boundary_pos > 0:
    overlap_text = overlap_text[:boundary_pos + 1].strip()

print()
print(f"Final overlap text ({len(overlap_text)} chars):")
print(repr(overlap_text))
print()
print("Full text:")
print(f"START: {overlap_text}")
print("...")
