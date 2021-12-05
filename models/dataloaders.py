import json
import torch
import random
from transformers import RobertaTokenizer, BartTokenizer
from torch.utils.data import Dataset

class ESNLIDataset(Dataset):
    def __init__(self, split, model, device):
        data = json.load(open(f'data/esnli/{split}.json'))
        if model == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        elif model == 'bart' or model == 'bart-expl':
            self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.data = data
        self.X = [example['premise'] + ' </s> ' + example['hypothesis'] for example in data]
        self.y_expl = [example['explanation'] if split=='train' else example['explanations'][0] for example in data]
        self.y_label = [example['label'] for example in data]
        self.model = model
        self.device = device

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        batch = self.tokenizer(self.X[idx], padding='longest', return_tensors='pt')
        batch['input_ids'] = batch['input_ids'].to(self.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.device)
        if self.model == 'roberta' or self.model == 'bart':
            batch['labels'] = torch.LongTensor(self.y_label[idx]).to(self.device)
        else:
            batch['classification_labels'] = torch.LongTensor(self.y_label[idx]).to(self.device)
            explanation_labels = self.tokenizer(self.y_expl[idx], padding='longest', return_tensors='pt')['input_ids']
            for i in range(len(explanation_labels)):
                if explanation_labels[i,1]==2:
                    explanation_labels[i] = -100
            batch['explanation_labels'] = explanation_labels.masked_fill(explanation_labels==self.tokenizer.pad_token_id, -100).to(self.device)
        return batch

    def shuffle(self):
        indices = list(range(len(self.X)))
        random.shuffle(indices)
        self.X = [self.X[i] for i in indices]
        self.y_label = [self.y_label[i] for i in indices]
        self.y_expl = [self.y_expl[i] for i in indices]


class ANLIDataset(Dataset):
    def __init__(self, level, split, include_backtranslation, model, device):
        if (level != 'R3' or split != 'train') and include_backtranslation:
            raise Exception('Backtranslation augmentation only available for R3 train set')
        data = json.load(open(f'data/anli/{level}/{split}.json'))
        if include_backtranslation:
            for lang in ['de', 'fr', 'ru', 'zh']:
                data.extend(json.load(open(f'data/anli/R3/augmentations/backtranslation/{lang}.json')))
        if model == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        elif model == 'bart' or model == 'bart-expl':
            self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.data = data
        self.X = [example['premise'] + ' </s> ' + example['hypothesis'] for example in data]
        self.y_expl = [example['explanation'] for example in data]
        self.y_label = [example['label'] for example in data]
        self.model = model
        self.device = device

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        batch = self.tokenizer(self.X[idx], padding='longest', return_tensors='pt')
        batch['input_ids'] = batch['input_ids'].to(self.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.device)
        if self.model == 'roberta' or self.model == 'bart':
            batch['labels'] = torch.LongTensor(self.y_label[idx]).to(self.device)
        else:
            batch['classification_labels'] = torch.LongTensor(self.y_label[idx]).to(self.device)
            explanation_labels = self.tokenizer(self.y_expl[idx], padding='longest', return_tensors='pt')['input_ids']
            for i in range(len(explanation_labels)):
                if explanation_labels[i,1]==2:
                    explanation_labels[i] = -100
            batch['explanation_labels'] = explanation_labels.masked_fill(explanation_labels==self.tokenizer.pad_token_id, -100).to(self.device)
        return batch

    def shuffle(self):
        indices = list(range(len(self.X)))
        random.shuffle(indices)
        self.X = [self.X[i] for i in indices]
        self.y_label = [self.y_label[i] for i in indices]
        self.y_expl = [self.y_expl[i] for i in indices]
