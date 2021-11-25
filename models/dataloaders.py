import json
import torch
import random
from transformers import RobertaTokenizer
from torch.utils.data import Dataset

class ESNLIDataset(Dataset):
    def __init__(self, split, device):
        data = json.load(open(f'data/esnli/{split}.json'))
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.X = [example['premise'] + ' </s> ' + example['hypothesis'] for example in data]
        self.y_expl = [example['explanation' if split=='train' else 'explanations'] for example in data]
        self.y_label = [example['label'] for example in data]
        self.device = device

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        batch = self.tokenizer(self.X[idx], padding='max_length', return_tensors='pt')
        batch['input_ids'] = batch['input_ids'].to(self.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.device)
        batch['labels'] = torch.LongTensor(self.y_label[idx]).to(self.device)
        return batch

    def shuffle(self):
        indices = list(range(len(self.X)))
        random.shuffle(indices)
        self.X = [self.X[i] for i in indices]
        self.y_label = [self.y_label[i] for i in indices]
        self.y_expl = [self.y_expl[i] for i in indices]
