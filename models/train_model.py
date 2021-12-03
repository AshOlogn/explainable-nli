import torch
import argparse
from dataloaders import ANLIDataset, ESNLIDataset
from expl_bart import BartForExplanatoryNLI
from transformers import RobertaForSequenceClassification, BartForSequenceClassification, AdamW
from os import mkdir, system
from os.path import isdir, join
from tqdm import tqdm
from utils import evaluate

def get_dirname(args):
    alpha_setting = f'_alpha-{args.alpha}' if args.model=='bart-expl' else ''
    return f'trained_models/{args.model}_{args.dataset}{alpha_setting}_epochs-{args.num_train_epochs}_bs-{args.batch_size}_lr-{args.learning_rate}'

def get_iter_indices(batch_size, length):
    indices = []
    i = 0
    while i < length:
        indices.append(i)
        i = min(i+batch_size, length)
    return indices

DATASET_TO_CLASS = {
    'esnli': (lambda split,model,device: ESNLIDataset(split, model, device)),
    'anli-1': (lambda split,model,device: ANLIDataset('R1', split, model, device)),
    'anli-2': (lambda split,model,device: ANLIDataset('R2', split, model, device)),
    'anli-3': (lambda split,model,device: ANLIDataset('R3', split, model, device))
}

def train(args):
    if args.model == 'bart':
        model = BartForSequenceClassification.from_pretrained('facebook/bart-base', num_labels=3).to(args.device)
    elif args.model == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3).to(args.device)
    elif args.model == 'bart-expl':
        model = BartForExplanatoryNLI.from_pretrained('facebook/bart-base', num_labels=3, alpha=args.alpha).to(args.device)
    model.train()

    dirname = get_dirname(args)
    if args.save_model:
        if isdir(dirname):
            if args.overwrite_old_model_dir:
                system(f'rm -r {dirname}')
            else:
                raise Exception(f'Model directory already exists, and overwriting isn\'t enabled')
        mkdir(dirname)

    train_dataset = DATASET_TO_CLASS[args.dataset]('train', args.model, args.device)
    dev_dataset = DATASET_TO_CLASS[args.dataset]('dev', args.model, args.device)

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
            loss = outputs[0]
            loss.backward()
            optimizer.step()

            steps += 1
            if args.validation_steps is not None and steps % (args.validation_steps)==0:
                model.eval()
                n_f1, e_f1, c_f1, m_f1, acc = evaluate(model, dev_dataset, args.batch_size)
                print(f'N F1 - {n_f1*100:.1f}%, E F1 - {e_f1*100:.1f}%, C F1 - {c_f1*100:.1f}, Mean F1 - {m_f1*100:.1f}, Accuracy - {acc*100:.1f}%')

                if args.save_model and acc > best_acc:
                    system(f'rm -r {dirname}/*')
                    fname = join(dirname, f'model_epoch-{e+1}_steps-{steps}_acc-{acc*100:.1f}.pt')
                    torch.save(model.state_dict(), fname)
                    best_acc = acc

                model.train()

            i = j

def main(args):
    if args.task=='train':
        train(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='train', choices=['train', 'predict'], required=False)
    parser.add_argument('--model', type=str, default='bart', choices=['bart', 'roberta', 'bart-expl'], required=False)
    parser.add_argument('--dataset', type=str, default='esnli', choices=['esnli', 'anli-1', 'anli-2', 'anli-3'], required=False)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--overwrite_old_model_dir", action="store_true")
    parser.add_argument('--load_path', type=str, default=None, required=False)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], required=False)
    parser.add_argument('--alpha', type=float, default=0.5, required=False)
    parser.add_argument('--learning_rate', type=float, default=3e-5, required=False)
    parser.add_argument('--batch_size', type=int, default=16, required=False)
    parser.add_argument('--num_train_epochs', type=int, default=3, required=False)
    parser.add_argument('--validation_steps', type=int, default=1000, required=False)

    args = parser.parse_args()
    main(args)
