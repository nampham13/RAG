import json
from collections import Counter

with open('loaders/test_pdf_blocks.json', 'r', encoding='utf-8') as f:
	data = json.load(f)

block_texts = []
block_details = []
for entry in data:
	for block in entry.get('blocks', []):
		# Handle both tuple format and dict format
		if isinstance(block, (list, tuple)) and len(block) >= 5:
			text = block[4].strip()
			bbox = (block[0], block[1], block[2], block[3]) if len(block) >= 4 else None
		elif isinstance(block, dict):
			text = block.get('text', '').strip()
			bbox = block.get('bbox')
		else:
			continue
        
		if text:
			block_texts.append(text)
			block_details.append({
				'text': text,
				'file': entry['file'],
				'page': entry['page'],
				'bbox': bbox
			})

counter = Counter(block_texts)
duplicates = [(text, count) for text, count in counter.items() if count > 1]

print(f"Total blocks: {len(block_texts)}")
print(f"Unique blocks: {len(counter)}")
print(f"Duplicate blocks: {len(duplicates)}")
print(f"Filter efficiency: {100 * (1 - len(block_texts) / 2044):.1f}% blocks removed\n")

print("=" * 80)
print("Top 20 duplicates with details:")
print("=" * 80)
for text, count in sorted(duplicates, key=lambda x: -x[1])[:20]:
	print(f"\n{count}x: {repr(text[:100])}")
	print(f"  Length: {len(text)} chars, {len(text.split())} words")
    
	# Show where it appears
	locations = []
	for detail in block_details:
		if detail['text'] == text:
			locations.append(f"{detail['file']} p.{detail['page']}")
	print(f"  Appears in: {', '.join(locations[:3])}" + (" ..." if len(locations) > 3 else ""))
