import json

with open('loaders/test_pdf_blocks.json', 'r', encoding='utf-8') as f:
	data = json.load(f)

# Find blocks with dots
for entry in data:
	for block in entry.get('blocks', []):
		if isinstance(block, (list, tuple)) and len(block) >= 5:
			text = block[4]
			if '...' in text and 'INTRODUCTION' in text.upper():
				print(f"File: {entry['file']}, Page: {entry['page']}")
				print(f"Text: {repr(text)}")
				print(f"Length: {len(text)}")
				print("=" * 80)
				break
	else:
		continue
	break
