from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, TaskType, get_peft_model

# def pre_data(sentence):
special_chars = {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '[SEP]', 'pad_token': '<pad>', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}

class FinetuneModel:
    def __init__(self, device, special_chars=None):
        self.device = device
        self.special_chars = special_chars
        self.model = self.tokenizer = None
        self.model_name = "rahular/varta-t5"
    
    def build_model(self):
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model = get_peft_model(model, peft_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)


        if self.special_chars:
            self.tokenizer.add_special_tokens(special_chars)
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)


if __name__ == '__main__':
    FTModel = FinetuneModel('cpu', special_chars)
    FTModel.build_model()

    # print(tokenizer.get_vocab())
    print(len(FTModel.tokenizer))
    # FTModel.tokenizer.add_special_tokens({'additional_special_tokens': ['<sos>', '<eos>']})

    print(len(FTModel.tokenizer))
    print(FTModel.tokenizer.get_added_vocab())
    # print(FT)

    print(FTModel.tokenizer('<s> निजी क्षेत्र में प्रदेश की 75 प्रतिशत नौकरियां हरियाणा के युवाओं के लिए आरक्षित की जाएगी। </s> <2hi>', '<2hi> <s> प्रदेश के युवाओं को निजी उद्योगों में 75 प्रतिशत आरक्षण देंगे। </s>',  add_special_tokens=False, return_tensors="pt", padding=True))
    # print(FTModel.tokenizer.decode(FTModel.tokenizer('<s> निजी क्षेत्र में प्रदेश की 75 प्रतिशत नौकरियां हरियाणा के युवाओं के लिए आरक्षित की जाएगी। </s> <2hi>', '<2hi> <s> प्रदेश के युवाओं को निजी उद्योगों में 75 प्रतिशत आरक्षण देंगे। </s>',  add_special_tokens=False, return_tensors="pt", padding=True)['input_ids'][0]))