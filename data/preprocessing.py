from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from indicnlp.tokenize import indic_tokenize
import random
from tqdm import tqdm


class ParaDataset(Dataset):
    def __init__(self, tokenizer, remove_percent=0.1, setting='train', keep_in_memory=True, only_str=False):
        # self.data = data
        self.setting = setting
        self.keep_in_memory = keep_in_memory
        self.tokenizer = tokenizer
        self.remove_percent = remove_percent
        self.only_str = only_str

        self.data = self.load_dataset()
        self.data = self.data[:50000]

        print(len(self.data))
        if only_str:
            return

        self.data = self.preprocess()

    def batch_input_processing(self, item):
        input_tokens = indic_tokenize.trivial_tokenize(item)
        input_tokens = random.sample(input_tokens, int((1 - self.remove_percent) * len(input_tokens)))
        random.shuffle(input_tokens)
        input_tokens = ' '.join(input_tokens)
        input_tokens = '<s> ' + input_tokens + ' </s>'
        target_tokens = '<s> ' + item + ' </s>'
        return input_tokens, target_tokens

    def preprocess(self):
        print("Processing data")

        inputs, targets = zip(*list(tqdm(map(self.batch_input_processing, self.data), total=len(self.data))))

        print(inputs[0])
        print(targets[0])

        batch_size = 1000
        inputs_output = {"input_ids": [], "attention_mask": [], "labels": []}



        for i in tqdm(range(0, len(inputs), batch_size), desc="Batching inputs"):
            batch = inputs[i:i+batch_size]
            target_batch = targets[i:i+batch_size]
            tokens = self.tokenizer(batch, text_target=target_batch, add_special_tokens=False)

            inputs_output["input_ids"].extend(tokens['input_ids'])
            inputs_output["attention_mask"].extend(tokens['attention_mask'])
            inputs_output["labels"].extend(tokens['labels'])

        print("Done")

        return inputs_output


    
    def __len__(self):
        if self.only_str:
            return len(self.data)
        return len(self.data['input_ids'])
    
    def __getitem__(self, idx):
        if self.only_str:
            return self.data[idx]

        input_ids = self.data['input_ids'][idx]
        attention_mask = self.data['attention_mask'][idx]
        labels = self.data['labels'][idx]


        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
    
    def load_dataset(self):
        res = []
        # targets = []
        if self.setting == 'train':
            dataset = load_dataset("ai4bharat/IndicParaphrase", 'hi', keep_in_memory=self.keep_in_memory)['train']
            for item in dataset:
                res.append(item['input'])
        elif self.setting == 'valid':
            dataset = load_dataset("ai4bharat/IndicParaphrase", 'hi', keep_in_memory=self.keep_in_memory)['validation']
            for item in dataset:
                res.append(item['input'])
        elif self.setting == 'test':
            dataset = load_dataset("ai4bharat/IndicParaphrase", 'hi', keep_in_memory=self.keep_in_memory)['test']
            for item in dataset:
                res.append(item['input'])
        
        return res

    
    def collate_fn(self, batch):
        input_ids, attention_mask, labels = zip(*batch)

        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return padded_input_ids, padded_attention_mask, padded_labels
        


def get_dataloader(dataset, batch_size=32, shuffle=True, collate_fn=None):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

