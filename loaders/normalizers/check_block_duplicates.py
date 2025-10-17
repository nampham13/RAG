import json
from collections import Counter

with open('loaders/test_pdf_blocks.json', 'r', encoding='utf-8') as f:
	data = json.load(f)

block_texts = []
for entry in data:
	for block in entry.get('blocks', []):
		# block[4] is the text content
		text = block[4].strip()
		if text:
			block_texts.append(text)

counter = Counter(block_texts)
duplicates = [(text, count) for text, count in counter.items() if count > 1]

print(f"Total blocks: {len(block_texts)}")
print(f"Duplicate blocks: {len(duplicates)}")
if duplicates:
	print("Top 10 duplicates:")
	for text, count in sorted(duplicates, key=lambda x: -x[1])[:10]:
		print(f"{count}x: {repr(text[:80])}")
else:
	print("No duplicate blocks found.")
