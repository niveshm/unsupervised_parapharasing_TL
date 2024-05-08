from data.preprocessing import ParaDataset, get_dataloader
from model import FinetuneModel
from transformers import Trainer, TrainingArguments, get_linear_schedule_with_warmup
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

FTModel = FinetuneModel(device)
FTModel.build_model()
dataset = ParaDataset(FTModel.tokenizer, setting='train')
print(len(dataset))
print(dataset[0])
train_dataloader = get_dataloader(dataset, batch_size=8, collate_fn=dataset.collate_fn)

val_dataset = ParaDataset(FTModel.tokenizer, setting='valid')

val_dataloader = get_dataloader(val_dataset, batch_size=8, shuffle=True, collate_fn=val_dataset.collate_fn)
print(len(val_dataset))


def train_model(model, epochs=10, lr=1e-4):
    optimizer = AdamW(model.parameters(), lr, weight_decay=0.01)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=epochs*len(train_dataloader))
    best_loss = np.inf


    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        model.train()
        train_loss = 0
        for i, (input_ids, attention_mask, labels) in tqdm(enumerate(train_dataloader)):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            # print(input_ids.shape, attention_mask.shape, labels.shape)
            optimizer.zero_grad()
            output = model(input_ids, attention_mask=attention_mask, labels=labels)
            output.loss.backward()
            optimizer.step()
            lr_scheduler.step()

            train_loss += output.loss.item()
            if i % 10 == 0:
                print(f"Epoch: {epoch}, Step: {i}, Loss: {output.loss.item()}")
            
        print("Train Loss: ", train_loss / len(train_dataloader))
        torch.save(model.state_dict(), f'./model_{epoch}.pt')
        
        print("Validation")
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (input_ids, attention_mask, labels) in tqdm(enumerate(val_dataloader)):
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                output = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += output.loss.item()
                if i % 10 == 0:
                    print(f"Epoch: {epoch}, Step: {i}, Val Loss: {output.loss.item()}")
            
            print("Val Loss: ", val_loss / len(val_dataloader))
        

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'./model_{epoch}.pt')
            print("Model saved")


train_model(FTModel.model, epochs=1, lr=1e-4)