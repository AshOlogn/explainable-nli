import json
from nltk.translate.bleu_score import corpus_bleu
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance

def corpus_bleu_score_outputs(fname):
    data = json.load(open(fname))
    references = [
        [example['explanation']] if 'explanation' in example else [example['explanations'][0]] 
        for example in data
    ]
    hypotheses = [example['generation'] for example in data]
    score = corpus_bleu(
        references,
        hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=None,
        auto_reweigh=False,
    )
    print(score)

def sentence_bert_embeddings(text, model):
    return model.encode(text)

def sentence_bert_outputs(fname):
    from tqdm import tqdm
    data = json.load(open(fname))
    references = [
        example['explanation'] if 'explanation' in example else example['explanations'][0] 
        for example in data
    ]
    hypotheses = [example['generation'] for example in data]

    model = SentenceTransformer('all-mpnet-base-v2')
    scores = []
    for r,h in tqdm(zip(references, hypotheses)):
        r_vector = sentence_bert_embeddings(r, model)
        h_vector = sentence_bert_embeddings(h, model)
        scores.append(1-distance.cosine(r_vector, h_vector))
    print(sum(scores)/len(scores))

import bert_score

def bertscore_outputs(fname):
    from tqdm import tqdm
    data = json.load(open(fname))
    references = [
        example['explanation'] if 'explanation' in example else example['explanations'][0] 
        for example in data
    ]
    hypotheses = [example['generation'] for example in data]

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
    print(sum(scores)/len(scores))

# corpus_bleu_score_outputs('trained_models/bart_expl_esnli_alpha-0.5_epochs-3_bs-4_lr-1e-05/predictions_esnli_dev.json')
# corpus_bleu_score_outputs('trained_models/bart-expl_anli-3_alpha-0.5_epochs-10_bs-4_lr-1e-05/predictions_anli-3_dev.json')
# corpus_bleu_score_outputs('trained_models/bart-expl_finetune_anli-3_alpha-0.5_epochs-10_bs-8_lr-1e-05/predictions_anli-3_dev.json')
# corpus_bleu_score_outputs('trained_models/bart-expl_anli-3_backtranslate_alpha-0.5_epochs-10_bs-4_lr-1e-05/predictions_anli-3_dev.json')

# sentence_bert_outputs('trained_models/bart_expl_esnli_alpha-0.5_epochs-3_bs-4_lr-1e-05/predictions_esnli_dev.json')
# sentence_bert_outputs('trained_models/bart-expl_anli-3_alpha-0.5_epochs-10_bs-4_lr-1e-05/predictions_anli-3_dev.json')
# sentence_bert_outputs('trained_models/bart-expl_finetune_anli-3_alpha-0.5_epochs-10_bs-8_lr-1e-05/predictions_anli-3_dev.json')
# sentence_bert_outputs('trained_models/bart-expl_anli-3_backtranslate_alpha-0.5_epochs-10_bs-4_lr-1e-05/predictions_anli-3_dev.json')

bertscore_outputs('trained_models/bart_expl_esnli_alpha-0.5_epochs-3_bs-4_lr-1e-05/predictions_esnli_dev.json')
bertscore_outputs('trained_models/bart-expl_anli-3_alpha-0.5_epochs-10_bs-4_lr-1e-05/predictions_anli-3_dev.json')
bertscore_outputs('trained_models/bart-expl_finetune_anli-3_alpha-0.5_epochs-10_bs-8_lr-1e-05/predictions_anli-3_dev.json')
bertscore_outputs('trained_models/bart-expl_anli-3_backtranslate_alpha-0.5_epochs-10_bs-4_lr-1e-05/predictions_anli-3_dev.json')