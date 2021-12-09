from transformers import BartTokenizer
import numpy as np
import json
from tqdm import tqdm
import bert_score
import os
import random

def edit_distance(s1, s2, tokenizer, normalize=True):
    t1 = tokenizer.encode(s1)[1:-1]
    t2 = tokenizer.encode(s2)[1:-1]
    memo = [[0] * len(t2) for i in range(len(t1))]

    #base cases
    memo[0][0] = int(t1[0]!=t2[0])-1
    for i in range(1,len(t1)):
        memo[i][0] = memo[i-1][0] if t1[i]!=t2[0] else -1

    for i in range(1,len(t2)):
        memo[0][i] = memo[0][i-1] if t1[0]!=t2[i] else -1

    for i in range(len(t1)):
        memo[i][0] += i+1

    for i in range(1,len(t2)):
        memo[0][i] += i+1

    # now just recurse
    for i in range(1, len(t1)):
        for j in range(1, len(t2)):
            if t1[i] == t2[j]:
               memo[i][j] = memo[i-1][j-1]
            else:
                 memo[i][j] = 1 + min(memo[i-1][j], memo[i][j-1], memo[i-1][j-1])
    else:
        return memo, memo[len(t1)-1][len(t2)-1]/(len(t1) if normalize else 1)

def translation_edit_distances(orig_fname, trans_fname):
    orig_data = json.load(open(orig_fname))
    orig_data = [example for example in orig_data if len(example['explanation']) > 0]
    trans_data = json.load(open(trans_fname))

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    neds = []
    for orig,trans in tqdm(list(zip(orig_data, trans_data))):
        neds.append(edit_distance(orig['premise'], trans['premise'], tokenizer, True)[1])
        neds.append(edit_distance(orig['hypothesis'], trans['hypothesis'], tokenizer, True)[1])
        neds.append(edit_distance(orig['explanation'], trans['explanation'], tokenizer, True)[1])

    print(f'{np.mean(neds)} ± {np.std(neds)}')

def translation_bertscores(orig_fname, trans_fname):
    orig_data = json.load(open(orig_fname))
    orig_data = [example for example in orig_data if len(example['explanation']) > 0]
    trans_data = json.load(open(trans_fname))

    hypotheses = []
    references = []
    for orig,trans in random.sample(list(zip(orig_data, trans_data)), 100):
        hypotheses.extend([
            trans['premise'], 
            trans['hypothesis'], 
            trans['explanation'] if 'explanation' in orig else orig['explanations'][0]
        ])
        references.extend([
            [orig['premise']], 
            [orig['hypothesis']], 
            [orig['explanation'] if 'explanation' in orig else orig['explanations'][0]]
        ])

    _, _, scores = bert_score.score(
        hypotheses,
        references,
        model_type=None,
        num_layers=None,
        verbose=False,
        idf=False,
        device=None,
        batch_size=64,
        nthreads=4,
        all_layers=False,
        lang='en',
        return_hash=False,
        rescale_with_baseline=True,
        baseline_path=None,
        use_fast_tokenizer=False
    )
    scores = scores.tolist()
    print(f'{np.mean(scores)} ± {np.std(scores)}')

def sample_translations(lang, k=10):
    orig_fname = 'data/anli/R3/train.json'
    trans_fname = f'data/anli/R3/augmentations/backtranslation/{lang}.json'
    orig_data = [example for example in json.load(open(orig_fname)) if len(example['explanation']) > 0]
    trans_data = json.load(open(trans_fname))
    samples = random.sample(list(zip(orig_data, trans_data)), k)

    text = '<html>\n'
    for orig,trans in samples:
        text += f"<h3>Id: {orig['id']}</h3>\n"
        text += f"<p><b>Premise</b>: {orig['premise']}</p>\n"
        text += f"<p><b>Translated Premise</b>: {trans['premise']}</p>\n"
        text += f"<p><b>Hypothesis</b>: {orig['hypothesis']}</p>\n"
        text += f"<p><b>Translated Hypothesis</b>: {trans['hypothesis']}</p>\n"
        text += f"<p><b>Explanation</b>: {orig['explanation']}</p>\n"
        text += f"<p><b>Translated Explanation</b>: {trans['explanation']}</p>\n"
    text += '</html>'

    if not os.path.isdir('data/anli/R3/augmentations/backtranslation/samples'):
        os.mkdir('data/anli/R3/augmentations/backtranslation/samples')

    outname = f'data/anli/R3/augmentations/backtranslation/samples/{lang}.html'
    open(outname, 'w').write(text)

# orig_fname = 'data/anli/R3/train.json'
# trans_fname = 'data/anli/R3/augmentations/backtranslation/fr.json'
# translation_bertscores(orig_fname, trans_fname)

# sample_translations('zh', k=10)