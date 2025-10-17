import json
import hashlib
import os

def hash_text(text):
	return hashlib.sha256(text.strip().encode('utf-8')).hexdigest()

def load_blocks(path):
	with open(path, 'r', encoding='utf-8') as f:
		data = json.load(f)
	blocks = []
	for entry in data:
		for block in entry.get('blocks', []):
			if isinstance(block, dict):
				text = block.get('text', '').strip()
			elif isinstance(block, (list, tuple)) and len(block) >= 5:
				text = block[4].strip()
			else:
				continue
			if text:
				blocks.append({
					'text': text,
					'file': entry.get('file'),
					'page': entry.get('page')
				})
	return blocks

def main():
	raw_path = os.path.join('loaders', 'test_pdf_blocks_raw.json')
	norm_path = os.path.join('loaders', 'test_pdf_blocks.json')
	if not os.path.exists(raw_path) or not os.path.exists(norm_path):
		print('Cần có cả file test_pdf_blocks_raw.json (trước normalize) và test_pdf_blocks.json (sau normalize)')
		return
	raw_blocks = load_blocks(raw_path)
	norm_blocks = load_blocks(norm_path)
	norm_hashes = set(hash_text(b['text']) for b in norm_blocks)
	lost_blocks = [b for b in raw_blocks if hash_text(b['text']) not in norm_hashes]
	print(f"Tổng số block bị mất hoàn toàn: {len(lost_blocks)}\n")
	for i, b in enumerate(lost_blocks[:50]):
		print(f"[{i+1}] File: {b['file']} | Page: {b['page']}\n{repr(b['text'])}\n{'-'*60}")
	if len(lost_blocks) > 50:
		print(f"... (còn {len(lost_blocks)-50} block nữa)")

if __name__ == '__main__':
	main()
