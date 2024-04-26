from datasets import load_dataset
from torch.utils.data import Dataset
from indicnlp.tokenize import indic_tokenize
import random
from tqdm import tqdm

# from ..model import FinetuneModel


class ParaDataset(Dataset):
    def __init__(self, tokenizer, max_len=1024, remove_percent=0.1, setting='train', device='cpu', keep_in_memory=False):
        # self.data = data
        self.max_len = max_len
        self.setting = setting
        inputs, targets = self.load_dataset()

        self.tokenizer = tokenizer
        self.remove_percent = remove_percent
        self.data, self.targets = self.preprocess(inputs, 'input')
        self.device = device
        self.keep_in_memory = keep_in_memory

        del inputs
        del targets

        # self.dataset = self.dataset.map(self.preprocess)
        # self.dataset.set_format(type='torch', columns=['input_tokens', 'target_tokens'])

    # def batch_input_processing(self, batch):
    #     res = []

    #     for item in batch:
    #         input_tokens = indic_tokenize.trivial_tokenize(item)
    #         input_tokens = random.sample(input_tokens, int((1 - self.remove_percent) * len(input_tokens)))
    #         random.shuffle(input_tokens)
    #         input_tokens = ' '.join(input_tokens)
    #         res.append(input_tokens)

    # def process_dataset(self, dataset):
    #     res = []
    #     for item in dataset:

    def preprocess(self, data, type='input'):

        if type == 'input':
            print("Processing data")
            ## multi thread this

            inputs = [indic_tokenize.trivial_tokenize(item) for item in data]
            inputs = [random.sample(item, int((1 - self.remove_percent) * len(item))) for item in inputs]
            for item in inputs:
                random.shuffle(item)
            
            inputs = [' '.join(item) for item in inputs]
            inputs = ['<sos> ' + item + ' <eos> <2hi>' for item in inputs]
            inputs = self.tokenizer(inputs, add_special_tokens=False, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")

            target = ['<2hi> <sos>' + item + ' <eos>' for item in data]
            target = self.tokenizer(target, add_special_tokens=False, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
            print("Done")

            return inputs, target
        else:
            target = ['<2hi> <sos>' + item + ' <eos>' for item in data]
            target = self.tokenizer(target, add_special_tokens=False, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")

            return target

        # # return self.tokenizer(data['input'], add_special_tokens=False, return_tensors="pt", padding="max_length", max_length=512, truncation=True)

        # # print(data['input'][0])
        # # print(len(data['input']))
        # # tmp_tokens = self.tokenizer(data['input'], add_special_tokens=False, return_tensors="pt")
        # # print(tmp_tokens['input_ids'].shape)
        # # exit()
        # input_text = data['input']
        # # print(input_text)
        # # target_text = data['target']

        # input_tokens = indic_tokenize.trivial_tokenize(input_text)

        # # print(input_tokens)
        # # print(len(input_tokens))
        # # print(self.remove_percent)
        # # print((1 - self.remove_percent) * len(input_tokens))

        # input_tokens = random.sample(input_tokens, int((1 - self.remove_percent) * len(input_tokens)))

        # random.shuffle(input_tokens)

        # input_tokens = ' '.join(input_tokens)

        # # print(input_tokens)
        
        # target_text = "<2hi> <sos>" + input_text + " <eos>"
        # input_text = "<sos> " + input_tokens + " <eos> <2hi>"

        # input_tokens = self.tokenizer(input_text, add_special_tokens=False, return_tensors="pt", max_length=self.max_len, padding='max_length', truncation=True)
        # # self.max_len = max(self.max_len, len(input_tokens['input_ids']))
        # target_tokens = self.tokenizer(target_text, add_special_tokens=False, return_tensors="pt", max_length=self.max_len, padding='max_length', truncation=True)
        # # print(target_tokens)
        # # print(target_tokens['input_ids'])
        # # print(target_tokens['input_ids'].shape)
        # # print(self.tokenizer.decode(target_tokens['input_ids'].squeeze(0)))

        # # self.max_len = max(self.max_len, len(target_tokens['input_ids'].squeeze(0)))

        # # exit()

        # return {'input_tokens': input_tokens, 'target_tokens': target_tokens}

        # # self.tokenizer(input_text, add_special_tokens=False, return_tensors="pt", padding="max_length", max_length=512, truncation=True)

        # # exit()
        # # निजी क्षेत्र में प्रदेश की 75 प्रतिशत नौकरियां हरियाणा के युवाओं के लिए आरक्षित की जाएगी।
        # # <2hi> <sos> निजी क्षेत्र में प्रदेश की 75 प्रतिशत नौकरियां हरियाणा के युवाओं के लिए आरक्षित की जाएगी। <eos>


    
    def __len__(self):
        return len(self.data['input_ids'])
    
    def __getitem__(self, idx):
        sample = {}
        # self.data['input_ids']
        sample['input_ids'] = self.data['input_ids'][idx].to(self.device)
        sample['attention_mask'] = self.data['attention_mask'][idx].to(self.device)
        sample['labels'] = self.targets['input_ids'][idx].to(self.device)
        # sample.to(self.device)

        return sample
    
    def load_dataset(self):
        res = []
        targets = []
        if self.setting == 'train':
            dataset = load_dataset("ai4bharat/IndicParaphrase", 'hi', keep_in_memory=self.keep_in_memory)['train']
            for item in dataset:
                res.append(item['input'])
                # targets.append(item['target'])
        elif self.setting == 'valid':
            dataset = load_dataset("ai4bharat/IndicParaphrase", 'hi', keep_in_memory=self.keep_in_memory)['validation']
            for item in dataset:
                res.append(item['input'])
                targets.append(item['target'])
        elif self.setting == 'test':
            dataset = load_dataset("ai4bharat/IndicParaphrase", 'hi', keep_in_memory=self.keep_in_memory)['test']
            for item in dataset:
                res.append(item['input'])
                targets.append(item['target'])
        
        return res, targets
        
        
        
        # return dataset

# dataset = load_dataset("ai4bharat/IndicParaphrase", 'hi')



# FTModel = FinetuneModel('cpu')
# FTModel.build_model()
# dataset = ParaDataset(FTModel.tokenizer, 'train')

