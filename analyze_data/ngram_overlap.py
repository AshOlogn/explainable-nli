import json
import os
from tqdm import tqdm
from nltk.tokenize import word_tokenize

def get_ngrams(tokens, n):
    ngrams = set([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
    return ngrams

def ngram_overlap(text1, text2, n):    
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    ngrams1 = get_ngrams(tokens1, n)
    ngrams2 = get_ngrams(tokens2, n)
    num_overlap = len(ngrams1 & ngrams2)

    # first one is precision, second one is recall
    return None if len(ngrams1|ngrams2)==0 else num_overlap/len(ngrams1|ngrams2)

def calculate_ngram_overlap(fname, dataset, n):
    data = json.load(open(fname))
    scores = [[], [], []]
    for example in tqdm(data):
        score = ngram_overlap(example['premise'], example['hypothesis'], n)
        if score is not None:
            scores[example['label']].append(score)

    if not os.path.isdir('results'):
        os.mkdir('results')

    with open(f'results/ngram_overlap_{dataset}-{n}.txt', 'w') as f:
        for s in scores[0]:
            f.write('0 ' + str(s) + '\n')

        for s in scores[1]:
            f.write('1 ' + str(s) + '\n')

        for s in scores[2]:
            f.write('2 ' + str(s) + '\n')

fname = 'data/esnli/train.json'
dataset = 'esnli'
calculate_ngram_overlap(fname, dataset, 1)
calculate_ngram_overlap(fname, dataset, 2)
calculate_ngram_overlap(fname, dataset, 3)

fname = 'data/anli/R3/train.json'
dataset = 'anli'
calculate_ngram_overlap(fname, dataset, 1)
calculate_ngram_overlap(fname, dataset, 2)
calculate_ngram_overlap(fname, dataset, 3)
