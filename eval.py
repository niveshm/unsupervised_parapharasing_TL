import torch
import pickle as pkl
from rouge import Rouge
from tqdm import tqdm

from data.metricprocess import ParaDataset
from model import FinetuneModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

dataset = ParaDataset()
print(len(dataset))
print("loaded data")

FTModel = FinetuneModel(device)
state_dict = torch.load('./ada/final_model_0.pt', map_location=torch.device('cpu'))
FTModel.build_model(state_dict=state_dict)
print("loaded model")

result = []


for i in tqdm(range(len(dataset))):
    input, target = dataset[i]
    # input.to(device)
    # target.to(device)

    res = FTModel.model.generate(FTModel.tokenizer(input, return_tensors="pt").input_ids, num_beams=4, max_length=30, min_length=1, early_stopping=True)

    res = FTModel.tokenizer.decode(res[0], skip_special_tokens=True)

    rouge = Rouge()

    scores = rouge.get_scores(res, target)
    result.append(scores)


with open('eval_result.pkl', 'wb') as f:
    pkl.dump(result, f)
