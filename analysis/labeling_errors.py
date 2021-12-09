import json
from collections import Counter

def accuracy(fname):
    data = json.load(open(fname))
    return sum([int(example['label']==example['prediction']) for example in data])/len(data)

def common_error_types(fname):
    data = json.load(open(fname))
    errors = Counter()
    for example in data:
        if example['label'] != example['prediction']:
            errors[(example['label'], example['prediction'])] += 1
    
    total = sum(list(errors.values()))
    ranked_errors = sorted(list(errors.items()), key=lambda x: x[1], reverse=True)
    for error,freq in ranked_errors:
        print(f"{error}: {freq/total*100:.1f}")

# common_error_types('trained_models/bart_expl_esnli_alpha-0.5_epochs-3_bs-4_lr-1e-05/predictions_esnli_dev.json')
# print()
# common_error_types('trained_models/bart-expl_anli-3_alpha-0.5_epochs-10_bs-4_lr-1e-05/predictions_anli-3_dev.json')

# print(accuracy('trained_models/bart-expl_anli-3_backtranslate_alpha-0.5_epochs-10_bs-4_lr-1e-05/predictions_anli-3_test.json'))