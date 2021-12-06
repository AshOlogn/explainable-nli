import torch
import argparse
from dataloaders import ANLIDataset, ESNLIDataset, ANLIExplanationDataset, ESNLIExplanationDataset
from expl_bart import BartForExplanatoryNLI
from transformers import RobertaForSequenceClassification, BartForSequenceClassification, AdamW
from transformers import BertTokenizer, BertForSequenceClassification
import os
from tqdm import tqdm
from utils import evaluate, get_predictions
from copy import deepcopy
import json

def get_dirname(args):
    dirname = f'trained_models/{args.model}'
    dirname += '_reduce-mean' if args.reduce=='mean' else ''
    dirname += '_finetune' if args.load_path is not None else ''
    dirname += f'_{args.dataset}'
    dirname += '_backtranslate' if args.use_backtranslation else ''
    dirname += f'_alpha-{args.alpha}' if args.model=='bart-expl' else '' 
    dirname += f'_epochs-{args.num_train_epochs}_bs-{args.batch_size}_lr-{args.learning_rate}'
    return dirname

def get_explanation_classifier_dirname(args):
    dirname = f'trained_models/expl-classifier'
    dirname += f'_{args.dataset}'
    dirname += f'_epochs-{args.num_train_epochs}_bs-{args.batch_size}_lr-{args.learning_rate}'
    return dirname

def get_iter_indices(batch_size, length):
    indices = []
    i = 0
    while i < length:
        indices.append(i)
        i = min(i+batch_size, length)
    return indices

DATASET_TO_CLASS = {
    'esnli': (lambda split,model,device: ESNLIDataset(split, model, device)),
    'anli-1': (lambda split,backtranslate,model,device: ANLIDataset('R1', split, backtranslate, model, device)),
    'anli-2': (lambda split,backtranslate,model,device: ANLIDataset('R2', split, backtranslate, model, device)),
    'anli-3': (lambda split,backtranslate,model,device: ANLIDataset('R3', split, backtranslate, model, device))
}

EXPLANATION_DATASET_TO_CLASS = {
    'esnli': (lambda split,device: ESNLIExplanationDataset(split, device)),
    'anli-1': (lambda split,device: ANLIExplanationDataset('R1', split, device)),
    'anli-2': (lambda split,device: ANLIExplanationDataset('R2', split, device)),
    'anli-3': (lambda split,device: ANLIExplanationDataset('R3', split, device)),
}

def train(args):
    if args.model == 'bart':
        model = BartForSequenceClassification.from_pretrained('facebook/bart-base', num_labels=3)
    elif args.model == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
    elif args.model == 'bart-expl':
        model = BartForExplanatoryNLI.from_pretrained('facebook/bart-base', num_labels=3, reduce=args.reduce, alpha=args.alpha)

    if args.load_path is not None:
        model.load_state_dict(torch.load(args.load_path))

    model.reduce = args.reduce
    model.alpha = args.alpha
    model.to(args.device)
    model.train()

    dirname = get_dirname(args)
    if args.save_model:
        if os.path.isdir(dirname):
            if args.overwrite_old_model_dir:
                os.system(f'rm -r {dirname}')
            else:
                raise Exception(f'Model directory already exists, and overwriting isn\'t enabled')
        os.mkdir(dirname)

    if args.dataset == 'esnli':
        train_dataset = DATASET_TO_CLASS[args.dataset]('train', args.model, args.device)
        dev_dataset = DATASET_TO_CLASS[args.dataset]('dev', args.model, args.device)
    else:
        train_dataset = DATASET_TO_CLASS[args.dataset]('train', args.use_backtranslation, args.model, args.device)
        dev_dataset = DATASET_TO_CLASS[args.dataset]('dev', False, args.model, args.device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    best_acc = 0
    steps = 0
    for e in range(args.num_train_epochs):
        print("***********************")
        print(f"Starting epoch {e+1}...")
        print("***********************")
        train_dataset.shuffle()

        indices = get_iter_indices(args.batch_size, len(train_dataset))
        for i in tqdm(indices):
            j = min(i+args.batch_size, len(train_dataset))
            batch = train_dataset[i:j]
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            steps += 1
            if args.validation_steps is not None and steps % (args.validation_steps)==0:
                model.eval()
                n_f1, e_f1, c_f1, m_f1, acc = evaluate(model, dev_dataset, args.batch_size)
                print(f'N F1 - {n_f1*100:.1f}%, E F1 - {e_f1*100:.1f}%, C F1 - {c_f1*100:.1f}, Mean F1 - {m_f1*100:.1f}, Accuracy - {acc*100:.1f}%')

                if args.save_model and acc > best_acc:
                    os.system(f'rm -r {dirname}/*')
                    fname = os.path.join(dirname, f'model_epoch-{e+1}_steps-{steps}_acc-{acc*100:.1f}.pt')
                    torch.save(model.state_dict(), fname)
                    best_acc = acc

                model.train()

            i = j

def train_explanation_classifier(args):
    model = BertForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=3)
    model.to(args.device)
    model.train()

    dirname = get_explanation_classifier_dirname(args)
    if args.save_model:
        if os.path.isdir(dirname):
            if args.overwrite_old_model_dir:
                os.system(f'rm -r {dirname}')
            else:
                raise Exception(f'Model directory already exists, and overwriting isn\'t enabled')
        os.mkdir(dirname)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    train_dataset = DATASET_TO_CLASS[args.dataset]('train', args.device)
    dev_dataset = DATASET_TO_CLASS[args.dataset]('dev', args.device)

    best_acc = 0
    steps = 0
    for e in range(args.num_train_epochs):
        print("***********************")
        print(f"Starting epoch {e+1}...")
        print("***********************")
        train_dataset.shuffle()

        indices = get_iter_indices(args.batch_size, len(train_dataset))
        for i in tqdm(indices):
            j = min(i+args.batch_size, len(train_dataset))
            batch = train_dataset[i:j]
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            steps += 1
            if args.validation_steps is not None and steps % (args.validation_steps)==0:
                model.eval()
                n_f1, e_f1, c_f1, m_f1, acc = evaluate(model, dev_dataset, args.batch_size)
                print(f'N F1 - {n_f1*100:.1f}%, E F1 - {e_f1*100:.1f}%, C F1 - {c_f1*100:.1f}, Mean F1 - {m_f1*100:.1f}, Accuracy - {acc*100:.1f}%')

                if args.save_model and acc > best_acc:
                    os.system(f'rm -r {dirname}/*')
                    fname = os.path.join(dirname, f'model_epoch-{e+1}_steps-{steps}_acc-{acc*100:.1f}.pt')
                    torch.save(model.state_dict(), fname)
                    best_acc = acc

                model.train()

            i = j


def predict(args):
    if args.model == 'bart':
        model = BartForSequenceClassification.from_pretrained('facebook/bart-base', num_labels=3)
    elif args.model == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
    elif args.model == 'bart-expl':
        model = BartForExplanatoryNLI.from_pretrained('facebook/bart-base', num_labels=3, reduce=args.reduce, alpha=args.alpha)

    if args.load_path is None:
        raise Exception('Need a path to load a model for prediction')
    else:    
        model.load_state_dict(torch.load(args.load_path))

    model.reduce = args.reduce
    model.alpha = args.alpha
    model.to(args.device)
    model.eval()

    if args.dataset == 'esnli':
        dataset = DATASET_TO_CLASS[args.dataset](args.predict_split, args.model, args.device)
    else:
        dataset = DATASET_TO_CLASS[args.dataset](args.predict_split, False, args.model, args.device)

    preds = get_predictions(model, dataset, args.batch_size, generate=True)
    preds['labels'] = preds['labels'].tolist()

    data = dataset.data
    results = []
    for i in range(len(preds['labels'])):
        example = deepcopy(data[i])
        example['prediction'] = preds['labels'][i]
        if args.model == 'bart-expl':
            example['generation'] = preds['generations'][i]
        results.append(example)

    results_fname = f'predictions_{args.dataset}_{args.predict_split}.json'
    results_fname = os.path.join(os.path.split(args.load_path)[0], results_fname)
    open(results_fname, 'w').write(json.dumps(results, indent=2))
    

def main(args):
    if args.task=='train':
        train(args)
    elif args.task == 'predict':
        predict(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='train', choices=['train', 'train-expl-clf', 'predict'], required=False)
    parser.add_argument('--model', type=str, default='bart', choices=['bart', 'roberta', 'bart-expl', 'bert-expl-clf'], required=False)
    parser.add_argument('--reduce', type=str, default='eos', choices=['eos', 'mean'], required=False)
    parser.add_argument('--dataset', type=str, default='esnli', choices=['esnli', 'anli-1', 'anli-2', 'anli-3'], required=False)
    parser.add_argument("--use_backtranslation", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--overwrite_old_model_dir", action="store_true")
    parser.add_argument('--load_path', type=str, default=None, required=False)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], required=False)
    parser.add_argument('--alpha', type=float, default=0.5, required=False)
    parser.add_argument('--learning_rate', type=float, default=3e-5, required=False)
    parser.add_argument('--batch_size', type=int, default=16, required=False)
    parser.add_argument('--num_train_epochs', type=int, default=3, required=False)
    parser.add_argument('--validation_steps', type=int, default=1000, required=False)
    parser.add_argument('--predict_split', type=str, default=None, choices=['train', 'dev', 'test'], required=False)

    args = parser.parse_args()
    main(args)
