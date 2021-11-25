import json
import pandas as pd

def format_esnli(split='train'):
    if split == 'train':
        df1 = pd.read_csv('data/esnli/raw/esnli_train_1.csv')
        df2 = pd.read_csv('data/esnli/raw/esnli_train_2.csv')
        df = pd.concat([df1, df2]).dropna(subset=['Sentence2'])
        data = [{
            'id': pair_id,
            'premise': sentence1,
            'hypothesis': sentence2,
            'label': 0 if label=='neutral' else 1 if label=='entailment' else 2,
            'explanation': explanation
        } for (pair_id, sentence1, sentence2, label, explanation) in 
            zip(df['pairID'], df['Sentence1'], df['Sentence2'], df['gold_label'], df['Explanation_1'])]
    else:
        df = pd.read_csv(f'data/esnli/raw/esnli_{split}.csv').dropna(subset=['Sentence2'])
        data = [{
            'id': pair_id,
            'premise': sentence1,
            'hypothesis': sentence2,
            'label': 0 if label=='neutral' else 1 if label=='entailment' else 2,
            'explanations': [explanation1, explanation2, explanation3]
        } for (pair_id, sentence1, sentence2, label, explanation1, explanation2, explanation3) in 
            zip(df['pairID'], df['Sentence1'], df['Sentence2'], df['gold_label'], 
            df['Explanation_1'], df['Explanation_1'], df['Explanation_1'])]
    open(f'data/esnli/{split}.json', 'w').write(json.dumps(data, indent=2))

def format_anli(split='train'):
    for level in ['R1', 'R2', 'R3']:
        raw_data = [json.loads(example) for example in open(f'data/anli/{level}/raw/{split}.jsonl').read().splitlines()]
        data = []
        for example in raw_data:
            data.append({
                'id': example['uid'],
                'premise': example['context'],
                'hypothesis': example['hypothesis'],
                'label': 0 if example['label']=='n' else 1 if example['label']=='e' else 2,
                'explanation': example['reason']
            })
        open(f'data/anli/{level}/{split}.json', 'w').write(json.dumps(data, indent=2))

# format_esnli('train')
# format_esnli('dev')
# format_esnli('test')

# format_anli('train')
# format_anli('dev')
# format_anli('test')