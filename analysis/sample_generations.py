import json
import os
import random

def sample_correct_classifications(fname):
    data = json.load(open(fname))
    sample_dir = os.path.join(os.path.split(fname)[0], 'correct_samples')
    if not os.path.isdir(sample_dir):
        os.mkdir(sample_dir)

    correct = [[], [], []]
    for example in data:
        if example['label']==example['prediction']:
            correct[example['label']].append(example)
    
    for i in range(3):
        sample = random.sample(correct[i], k=10)
        open(os.path.join(sample_dir, f'sample-{i}.json'), 'w').write(json.dumps(sample, indent=2))

def sample_incorrect_classifications(fname):
    data = json.load(open(fname))
    sample_dir = os.path.join(os.path.split(fname)[0], 'incorrect_samples')
    if not os.path.isdir(sample_dir):
        os.mkdir(sample_dir)

    correct = {(0,1):[], (1,0):[], (0,2):[], (2,0):[], (1,2):[], (2,1):[]}
    for example in data:
        if example['label'] != example['prediction']:
            correct[(example['label'], example['prediction'])].append(example)

    for k in correct:
        sample = random.sample(correct[k], 10)
        open(os.path.join(sample_dir, f'sample-{k[0]}-{k[1]}.json'), 'w').write(json.dumps(sample, indent=2))

# sample_correct_classifications('trained_models/bart_expl_esnli_alpha-0.5_epochs-3_bs-4_lr-1e-05/predictions_esnli_dev.json')
# sample_correct_classifications('trained_models/bart-expl_anli-3_alpha-0.5_epochs-10_bs-4_lr-1e-05/predictions_anli-3_dev.json')
# sample_incorrect_classifications('trained_models/bart_expl_esnli_alpha-0.5_epochs-3_bs-4_lr-1e-05/predictions_esnli_dev.json')
# sample_incorrect_classifications('trained_models/bart-expl_anli-3_alpha-0.5_epochs-10_bs-4_lr-1e-05/predictions_anli-3_dev.json')