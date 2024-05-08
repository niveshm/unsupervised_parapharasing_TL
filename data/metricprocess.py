from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from indicnlp.tokenize import indic_tokenize
import random
from tqdm import tqdm


class ParaDataset(Dataset):
    def __init__(self, keep_in_memory=True):
        # self.data = data
        self.keep_in_memory = keep_in_memory

        self.data, self.target = self.load_dataset()
        self.data = self.data[:100]
        self.target = self.target[:100]

        print(len(self.data))


    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
    def load_dataset(self):
        res = []
        targets = []
        dataset = load_dataset("ai4bharat/IndicParaphrase", 'hi', keep_in_memory=self.keep_in_memory)['test']
        for item in dataset:
            res.append(item['input'])
            targets.append(item['target'])
        # if self.setting == 'train':
        #     dataset = load_dataset("ai4bharat/IndicParaphrase", 'hi', keep_in_memory=self.keep_in_memory)['train']
        #     for item in dataset:
        #         res.append(item['input'])
        # elif self.setting == 'valid':
        #     dataset = load_dataset("ai4bharat/IndicParaphrase", 'hi', keep_in_memory=self.keep_in_memory)['validation']
        #     for item in dataset:
        #         res.append(item['input'])
        # elif self.setting == 'test':
        #     dataset = load_dataset("ai4bharat/IndicParaphrase", 'hi', keep_in_memory=self.keep_in_memory)['test']
        #     for item in dataset:
        #         res.append(item['input'])
        
        return res, targets

    
    # def collate_fn(self, batch):
    #     input_ids, attention_mask, labels = zip(*batch)

    #     padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
    #     padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    #     padded_labels = pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)
    #     return padded_input_ids, padded_attention_mask, padded_labels

