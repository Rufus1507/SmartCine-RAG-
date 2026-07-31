import sys, json, os
sys.stdout.reconfigure(encoding='utf-8')

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Traditional RAG
trad_path = os.path.join(root, 'eval', 'traditional_results_raw.json')
hq_path = os.path.join(root, 'eval', 'hq_results_raw.json')

for label, path in [('Traditional RAG', trad_path), ('CineBot V3', hq_path)]:
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                results = json.load(f)
            last = results[-1] if results else None
            print(f"[{label}] {len(results)}/100 cau hoan thanh")
            if last:
                print(f"  Cau cuoi: {last['id']} | latency={last.get('latency_s','?')}s")
        except Exception as e:
            print(f"[{label}] Loi doc file: {e}")
    else:
        print(f"[{label}] Chua co file ket qua")
