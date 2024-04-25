from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# def pre_data(sentence):

class FinetuneModel:
    def __init__(self, device, special_chars=['<sos>', '<eos>']):
        self.device = device
        self.special_chars = special_chars
        self.model = self.tokenizer = None
    
    def build_model(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBART")
        self.tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)
        # for char in self.special_chars:
            # self.tokenizer.add_special_tokens({'additional_special_tokens': [char]})
            # self.tokenizer.get
        self.tokenizer.add_special_tokens({'additional_special_tokens': self.special_chars})
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)


if __name__ == '__main__':
    FTModel = FinetuneModel('cpu')
    FTModel.build_model()

    # print(tokenizer.get_vocab())
    print(len(FTModel.tokenizer))
    FTModel.tokenizer.add_special_tokens({'additional_special_tokens': ['<sos>', '<eos>']})

    print(len(FTModel.tokenizer))
    print(FTModel.tokenizer.get_added_vocab())

    print(FTModel.tokenizer('<s> निजी क्षेत्र में प्रदेश की 75 प्रतिशत नौकरियां हरियाणा के युवाओं के लिए आरक्षित की जाएगी। </s> <2hi>', '<2hi> <s> प्रदेश के युवाओं को निजी उद्योगों में 75 प्रतिशत आरक्षण देंगे। </s>',  add_special_tokens=False, return_tensors="pt", padding=True))

    # bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
    # eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
    # # pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")
    # # print(tokenizer._convert_token_to_id("<s>"))
    # # print(tokenizer._convert_id_to_token(3))
    # # print(bos_id, eos_id, pad_id)
    # inp = tokenizer('<s> निजी क्षेत्र में प्रदेश की 75 प्रतिशत नौकरियां हरियाणा के युवाओं के लिए आरक्षित की जाएगी। </s> <2hi>',  add_special_tokens=False, return_tensors="pt", padding=True)
    # out = tokenizer('<2hi> <s> प्रदेश के युवाओं को निजी उद्योगों में 75 प्रतिशत आरक्षण देंगे। </s>', add_special_tokens=False, return_tensors="pt", padding=True)

    # res = model(input_ids=inp['input_ids'], decoder_input_ids=out['input_ids'][:, :-1], labels=out['input_ids'][:, 1:])
    # print(res)
    # # ['[CLS]', '<s>', '▁निजी', '▁क्षेत्र', '▁में', '▁प्रदेश', '▁की', '▁75', '▁प्रतिशत', '▁नौकर', 'ियां', '▁हरियाणा', '▁के', '▁युवाओं', '▁के', '▁लिए', '▁आरक्षित', '▁की', '▁जाएगी', '।', '</s>', '[SEP]']
    # # print(tokenizer.convert_ids_to_tokens(res['input_ids']))
    # # print(tokenizer('<s> निजी क्षेत्र में प्रदेश की 75 प्रतिशत नौकरियां हरियाणा के युवाओं के लिए आरक्षित की जाएगी। </s> <2hi>'))