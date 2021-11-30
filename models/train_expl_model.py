import torch
import argparse
from dataloaders import ANLIDataset, ESNLIDataset
from expl_bart import BartForExplanatoryNLI
from transformers import AdamW
from tqdm import tqdm
from utils import evaluate

def get_dirname(args):
    return f'trained_models/expl-bart_base_epochs-{args.num_train_epochs}_bs-{args.batch_size}_lr-{args.learning_rate}'

def get_iter_indices(batch_size, length):
    indices = []
    i = 0
    while i < length:
        indices.append(i)
        i = min(i+batch_size, length)
    return indices

DATASET_TO_CLASS = {
    'esnli': (lambda split,device: ESNLIDataset(split, 'bart-expl', device)),
    'anli-1': (lambda split,device: ANLIDataset('R1', split, 'bart-expl', device)),
    'anli-2': (lambda split,device: ANLIDataset('R2', split, 'bart-expl', device)),
    'anli-3': (lambda split,device: ANLIDataset('R3', split, 'bart-expl', device))
}

def train(args):
    model = BartForExplanatoryNLI.from_pretrained('facebook/bart-base', num_labels=3).to(args.device)
    train_dataset = DATASET_TO_CLASS[args.dataset]('train', args.device)
    dev_dataset = DATASET_TO_CLASS[args.dataset]('dev', args.device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

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
            batch['alpha'] = args.alpha
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
                model.train()

            i = j

def main(args):
    if args.task=='train':
        train(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='train', choices=['train', 'predict'], required=False)
    parser.add_argument('--dataset', type=str, default='esnli', choices=['esnli', 'anli-1', 'anli-2', 'anli-3'], required=False)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], required=False)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--learning_rate', type=float, default=3e-5, required=False)
    parser.add_argument('--batch_size', type=int, default=16, required=False)
    parser.add_argument('--num_train_epochs', type=int, default=3, required=False)
    parser.add_argument('--validation_steps', type=int, default=1000, required=False)

    args = parser.parse_args()
    main(args)
