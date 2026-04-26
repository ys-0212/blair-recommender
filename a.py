with open('src/stage6_features/feature_builder.py', encoding='utf-8') as f:
    content = f.read()
if 'get_batch_scores' in content:
    print('BAD: still using get_batch_scores')
    # Find the line
    for i, line in enumerate(content.split('\n')):
        if 'get_batch_scores' in line or 'get_scores' in line:
            print(f'Line {i+1}: {line.strip()}')
elif 'get_scores' in content:
    print('GOOD: using get_scores')
else:
    print('No BM25 scoring found')
