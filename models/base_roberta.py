import torch
import argparse
from dataloaders import ANLIDataset, ESNLIDataset
from transformers import RobertaForSequenceClassification, AdamW
from os import mkdir, system
from os.path import isdir
from tqdm import tqdm

def get_dirname(args):
    return f'trained_models/roberta_base_epochs-{args.num_train_epochs}_bs-{args.batch_size}_lr-{args.learning_rate}'

def get_iter_indices(batch_size, length):
    indices = []
    i = 0
    while i < length:
        indices.append(i)
        i = min(i+batch_size, length)
    return indices

DATASET_TO_CLASS = {
    'esnli': (lambda split,device: ESNLIDataset(split, 'roberta', device)),
    'anli-1': (lambda split,device: ANLIDataset('R1', split, 'roberta', device)),
    'anli-2': (lambda split,device: ANLIDataset('R2', split, 'roberta', device)),
    'anli-3': (lambda split,device: ANLIDataset('R3', split, 'roberta', device))
}

def train(args):
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3).to(args.device)
    model.train()

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
                # torch.save(model.state_dict(), f'{dirname}/model_epoch-{e+1}_steps-{steps}_acc-{acc*100:.1f}.pt')
                model.train()

            i = j

@torch.no_grad()
def get_predictions(model, dataset, batch_size):
    model.eval()
    preds = []
    indices = get_iter_indices(batch_size, len(dataset))
    for i in indices:
        j = min(i+batch_size, len(dataset))
        batch = dataset[i:j]
        del batch['labels']
        logits = model(**batch)[0]
        preds.append(torch.argmax(logits, dim=-1).to('cpu'))

    preds = torch.cat(preds)
    return preds

def tensor_accuracy(preds, targets):
    return torch.sum(preds.eq(targets).int()).item()/len(preds)

def tensor_binary_precision(preds, targets):
    num_labeled_positive = torch.sum(preds).item()
    if num_labeled_positive == 0:
        return 0
    num_correctly_labeled_positive = torch.sum(preds * preds.eq(targets).int()).item()
    return num_correctly_labeled_positive/num_labeled_positive 

def tensor_binary_recall(preds, targets):
    num_truly_positive = torch.sum(targets).item()
    if num_truly_positive==0:
        return 0
    num_correctly_labeled_positive = torch.sum(preds * preds.eq(targets).int()).item()
    return num_correctly_labeled_positive/num_truly_positive

def tensor_binary_f1(preds, targets):
    precision = tensor_binary_precision(preds, targets)
    recall = tensor_binary_recall(preds, targets)
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

def multiclass_f1(preds, targets):
    scores = []
    for i in range(3):
        scores.append(tensor_binary_f1(preds.eq(i).int(), targets.eq(i).int()))
    scores.append(sum(scores)/3)
    return scores

def evaluate(model, dataset, batch_size):
    preds = get_predictions(model, dataset, batch_size)
    targets = torch.LongTensor(dataset.y_label)
    n_f1, e_f1, c_f1, m_f1 =  multiclass_f1(preds, targets)
    acc = tensor_accuracy(preds, targets)
    return n_f1, e_f1, c_f1, m_f1, acc

def main(args):
    if args.task=='train':
        train(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='train', choices=['train', 'predict'], required=False)
    parser.add_argument('--dataset', type=str, default='esnli', choices=['esnli', 'anli-1', 'anli-2', 'anli-3'], required=False)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], required=False)
    parser.add_argument('--learning_rate', type=float, default=3e-5, required=False)
    parser.add_argument('--batch_size', type=int, default=16, required=False)
    parser.add_argument('--num_train_epochs', type=int, default=3, required=False)
    parser.add_argument('--validation_steps', type=int, default=1000, required=False)

    args = parser.parse_args()
    main(args)