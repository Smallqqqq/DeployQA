
from transformers import RobertaTokenizer, RobertaForQuestionAnswering
import torch
tokenizer = RobertaTokenizer.from_pretrained('/mdoel')
model = RobertaForQuestionAnswering.from_pretrained('/model')

marked_text='I. Download the zip file http://www.mongodb.org/downloads'
tokenized_text = tokenizer.tokenize(marked_text)
print (tokenized_text)



