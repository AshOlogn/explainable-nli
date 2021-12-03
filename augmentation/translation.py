from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from multiprocessing import Process
from tqdm import tqdm
import os
import json
import gc

def get_iter_indices(batch_size, length):
    indices = []
    i = 0
    while i < length:
        indices.append(i)
        i = min(i+batch_size, length)
    return indices

def translate(texts, src_lang, tgt_lang, batch_size, device):
    tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}")
    model = AutoModelForSeq2SeqLM.from_pretrained(f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}").to(device)
    model.eval()

    outputs = []
    for i in tqdm(get_iter_indices(batch_size, len(texts))):
        j = min(i+batch_size, len(texts))
        input_ids = tokenizer(texts[i:j], padding='longest', return_tensors='pt')['input_ids'].to(device)
        output_ids = model.generate(input_ids, max_length=256, do_sample=True, top_p=0.9)
        outputs.extend([tokenizer.decode(output_ids[i], skip_special_tokens=True) for i in range(len(output_ids))])    

    del tokenizer
    del model
    gc.collect()
    return outputs

def backtranslate(texts, src_lang, tgt_lang, batch_size, device):
    texts1 = translate(texts, src_lang, tgt_lang, batch_size, device)
    texts2 = translate(texts1, tgt_lang, src_lang, batch_size, device)
    return texts2

def augment_anli3_lang(lang, batch_size, device):
    data = [example for example in json.load(open('data/anli/R3/train.json')) if len(example['explanation']) > 0]
    ids = [example['id'] for example in data]
    labels = [example['label'] for example in data]
    premises = [example['premise'] for example in data]
    hypotheses = [example['hypothesis'] for example in data]
    explanations = [example['explanation'] for example in data]
    texts = premises + hypotheses + explanations
    aug_texts = backtranslate(texts, 'en', lang, batch_size)
    aug_premises = aug_texts[:len(data)]
    aug_hypotheses = aug_texts[len(data):2*len(data)]
    aug_explanations = aug_texts[2*len(data):]
    augmented_data = [{
        'id': f'backtranslate-{lang}_{id}',
        'premise': premise,
        'hypothesis': hypothesis,
        'label': label,
        'explanation': explanation
    } for id,premise,hypothesis,label,explanation in zip(ids, labels, aug_premises, aug_hypotheses, aug_explanations)]

    if not os.path.isdir('data/anli/R3/augmentations'):
        os.mkdir('data/anli/R3/augmentations')

    if not os.path.isdir('data/anli/R3/augmentations/backtranslation'):
        os.mkdir('data/anli/R3/augmentations/backtranslation')

    fname = f'data/anli/R3/augmentations/backtranslation/{lang}.json'
    open(fname, 'w').write(json.dumps(augmented_data, indent=2))

def augment_anli3(batch_size):
    process_fr = Process(target=augment_anli3_lang, args=('fr', batch_size, 'cuda:0'))
    process_de = Process(target=augment_anli3_lang, args=('de', batch_size, 'cuda:1'))
    process_ru = Process(target=augment_anli3_lang, args=('ru', batch_size, 'cuda:2'))
    process_zh = Process(target=augment_anli3_lang, args=('zh', batch_size, 'cuda:3'))
    process_fr.start()
    process_de.start()
    process_ru.start()
    process_zh.start()

augment_anli3(batch_size=16)