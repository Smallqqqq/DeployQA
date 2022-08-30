from transformers import RobertaTokenizer, RobertaForMaskedLM

tokenizer = RobertaTokenizer.from_pretrained('/mdoel')
model = RobertaForMaskedLM.from_pretrained('/model')

marked_text = 'I. Download the zip file http://www.mongodb.org/downloads'
tokenized_text = tokenizer.tokenize(marked_text)
print(tokenized_text)
