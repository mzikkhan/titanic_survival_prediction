import json

notebook_path = '/Users/zaedkhan/Desktop/data_572/project/project.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('Accuracy Comparison (Test Set)' in line for line in source):
            # Find where to insert plt.ylim(0.7, 1.0)
            for i, line in enumerate(source):
                if 'plt.ylabel("Accuracy")' in line:
                    source.insert(i, 'plt.ylim(0.7, 1.0)\n')
                    break
            break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
