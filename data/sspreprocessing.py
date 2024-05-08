from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class ParaDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        print("Loading dataset")
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data, self.data_rev = self.load_dataset()

    
    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):

        if np.random.rand() < 0.1:
            input_ids = self.data_rev['input_ids'][idx]
            target = self.data_rev['labels'][idx]
            attention_mask = self.data_rev['attention_mask'][idx]
            

            return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long), torch.tensor(target, dtype=torch.long)
        
        input_ids = self.data['input_ids'][idx]
        target = self.data['labels'][idx]
        attention_mask = self.data['attention_mask'][idx]
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long), torch.tensor(target, dtype=torch.long)
    

    def load_dataset(self):
        data = pd.read_csv(self.data_path)
        print(len(data))

        input = data['data'].tolist()
        target = data['result'].tolist()
        # print(len(input), len(target))
        # print(input)
        batch_size = 1000

        inputs_output = {"input_ids": [], "attention_mask": [], "labels": []}
        inputs_output_rev = {"input_ids": [], "attention_mask": [], "labels": []}

        for i in tqdm(range(0, len(input), batch_size)):
            batch = input[i:i+batch_size]
            target_batch = target[i:i+batch_size]
            tokens = self.tokenizer(batch, text_target=target_batch, add_special_tokens=False)
            tokens_rev = self.tokenizer(target_batch, text_target=batch, add_special_tokens=False)
            inputs_output["input_ids"].extend(tokens['input_ids'])
            inputs_output["attention_mask"].extend(tokens['attention_mask'])
            inputs_output["labels"].extend(tokens['labels'])

            inputs_output_rev["input_ids"].extend(tokens_rev['input_ids'])
            inputs_output_rev["attention_mask"].extend(tokens_rev['attention_mask'])
            inputs_output_rev["labels"].extend(tokens_rev['labels'])
        

        # tokens = self.tokenizer(input, text_target=target, add_special_tokens=False)
        # tokens_rev = self.tokenizer(target, text_target=input, add_special_tokens=False)
        return inputs_output, inputs_output_rev
    
    def collate_fn(self, batch):
        input_ids, attention_mask, labels = zip(*batch)

        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return padded_input_ids, padded_attention_mask, padded_labels