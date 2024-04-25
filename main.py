from data.preprocessing import ParaDataset
from model import FinetuneModel
from transformers import Trainer, TrainingArguments

FTModel = FinetuneModel('cpu')
FTModel.build_model()
dataset = ParaDataset(FTModel.tokenizer, setting='train')
print(dataset.max_len)
val_dataset = ParaDataset(FTModel.tokenizer, setting='valid')

# print(dataset.dataset[0])
# print("TEST===============")
# print(dataset.test[0])

train_args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=10,
    eval_steps=10,
    save_total_limit=2,
    eval_accumulation_steps=2,
    evaluation_strategy='steps',
    output_dir='./output',
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    do_predict=True,
    remove_unused_columns=False,
    weight_decay=0.01,
    learning_rate=5e-5,
)

trainer = Trainer(
    model=FTModel.model,
    args=train_args,
    train_dataset=dataset,
    eval_dataset=val_dataset,
    # data_collator=dataset.collate_fn,
)

trainer.train()