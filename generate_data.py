from indicnlp.tokenize import indic_tokenize
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from data.preprocessing import ParaDataset
from model import FinetuneModel

def get_last_word(decoder_input, tokenizer):
    # print(decoder_input)
    res = tokenizer.decode(decoder_input)
    # print(res)
    res = indic_tokenize.trivial_tokenize(res)
    return res[-1]

def get_full_block(source_squence):
    # source_squence_tokens = tokenizer(source_squence)

    # source_squence_tokens = tokenizer.convert_ids_to_tokens(source_squence_tokens['input_ids'])

    source_squence_tokens = indic_tokenize.trivial_tokenize(source_squence)

    source_squence_tokens = ['<pad>'] + source_squence_tokens + ['</s>']
    # print(source_squence_tokens)

    full_block = {}
    for i in range(len(source_squence_tokens) - 1):
        full_block[source_squence_tokens[i]] = source_squence_tokens[i+1]

    return full_block


def get_active_block(full_block, probability):
    active_block = {}
    for key, value in full_block.items():
        if np.random.rand() < probability:
            active_block[key] = value
    return active_block


def generate_text(model, tokenizer, input_sentence, active_block, max_length=100):
    model.eval()
    encoder_input_ids = tokenizer(input_sentence, return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)

    s_token_id = tokenizer.convert_tokens_to_ids('<pad>')
    decoder_input_ids = torch.tensor([s_token_id], dtype=torch.long, device=model.device).unsqueeze(0)

    generated_tokens = []
    generated_tokens_id = []
    current_token = '<pad>'

    for _ in range(max_length):
        # print("Start")
        outputs = model(input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.topk(next_token_logits, k=2, dim=-1).indices

        # print(next_token_id.squeeze()[0].unsqueeze(0))
        # break
        next_token = [0, 0]
        # print(next_token_id)

        next_token[0] = tokenizer.decode(next_token_id.squeeze()[0].unsqueeze(0).tolist())
        next_token[1] = tokenizer.decode(next_token_id.squeeze()[1].unsqueeze(0).tolist())

        # next_token_id = next_token_id[0].tolist()
        # words = indic_tokenize.trivial_tokenize(next_token)
        words = next_token
        # print(words)
        

        # if words[0] == tokenizer.eos_token:
        #     break
        
        
        # print(next_token)
        # break
        # Here is the issue
        # if next_token:
        #     print(next_token)
        #     # if active_block[current_token] == next_token[0]:
        #     #     next
            
        #     continue  
        # print(next_token)
        if active_block.get(current_token, None) == words[0]:
            # print("here")
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id[0][1].unsqueeze(0).unsqueeze(0)], dim=-1)

            generated_tokens.append(next_token[1])
            current_token = next_token[1]
            # generated_tokens_id.append(next_token_id[1])
        
        else:
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id[0][0].unsqueeze(0).unsqueeze(0)], dim=-1)

            generated_tokens.append(next_token[0])
            current_token = next_token[0]
            # generated_tokens_id.append(next_token_id[0])
        
        
        if current_token == tokenizer.eos_token:
            break

        current_token = get_last_word(decoder_input_ids[0], tokenizer)
        # print(current_token)
        
        # print('')

    # print(generated_tokens)
    # print(generated_tokens_id)
    # # print(tokenizer.decode([generated_tokens_id]))
    # print(decoder_input_ids)
    return tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

FTModel = FinetuneModel(device)
state_dict = torch.load('./model_0.pt')
FTModel.build_model(state_dict=state_dict)
print("loaded model")
dataset = ParaDataset(FTModel.tokenizer, setting='train', only_str=True)

data = []
result = []
print("Generating Data")

try: 

    for i in tqdm(range(len(dataset))):
        block = get_active_block(get_full_block(dataset[i]), 0.5)
        data.append(dataset[i])
        # print(dataset[i])
        inp = "<s> " + dataset[i] + " </s>"
        res = generate_text(FTModel.model, FTModel.tokenizer, inp, block)
        # print(res)
        result.append(res)
        # break

except KeyboardInterrupt:
    print("Keyboard Interrupt")
    print("Saving Data")
    df = pd.DataFrame({'data': data, 'result': result})
    df.to_csv('data.csv', index=False)
    print("Data Saved")
    exit()

## if key board interruption save the data



## save data and result to csv
df = pd.DataFrame({'data': data, 'result': result})
df.to_csv('data.csv', index=False)

print("Done")

