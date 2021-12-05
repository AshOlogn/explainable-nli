import torch
from expl_bart import BartForExplanatoryNLI

def get_iter_indices(batch_size, length):
    indices = []
    i = 0
    while i < length:
        indices.append(i)
        i = min(i+batch_size, length)
    return indices

@torch.no_grad()
def get_generations(model, batch, tokenizer):
    output_ids = model.generate(batch['input_ids'], max_length=50, num_beams=5)
    texts = [tokenizer.decode(output_ids[i], skip_special_tokens=True) for i in range(len(output_ids))]
    return texts

@torch.no_grad()
def get_predictions(model, dataset, batch_size, generate=False):
    model.eval()
    labels = []
    gens = []
    #indices = get_iter_indices(batch_size, len(dataset))
    indices = get_iter_indices(batch_size, 10)
    for i in indices:
        j = min(i+batch_size, len(dataset))
        batch = dataset[i:j]
        if isinstance(model, BartForExplanatoryNLI):
            del batch['classification_labels']
            del batch['explanation_labels']
            logits = model(**batch).classification_logits
            if generate:
                gens.extend(get_generations(model, batch, dataset.tokenizer))
        else:
            del batch['labels']
            logits = model(**batch).logits
        labels.append(torch.argmax(logits, dim=-1).to('cpu'))

    labels = torch.cat(labels)
    predictions = {'labels': labels}
    if generate and isinstance(model, BartForExplanatoryNLI):
        predictions['generations'] = gens    
    return predictions

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

def evaluate(model, dataset, batch_size, generate=False):
    predictions = get_predictions(model, dataset, batch_size, generate)
    targets = torch.LongTensor(dataset.y_label)
    n_f1, e_f1, c_f1, m_f1 =  multiclass_f1(predictions['labels'], targets)
    acc = tensor_accuracy(predictions['labels'], targets)
    return n_f1, e_f1, c_f1, m_f1, acc
