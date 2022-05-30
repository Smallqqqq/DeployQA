import transformers
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

f = open("train.txt")
lines = f.read()
print(lines)
lines = [lines]
f.close()
count_vectorizer = CountVectorizer(lowercase=True,max_features=12000)
fit=count_vectorizer.fit_transform(lines)
name = count_vectorizer.get_feature_names()
voca =count_vectorizer.vocabulary_
stop = count_vectorizer.get_stop_words()
count=fit.toarray().sum(axis=0)
count_dict=dict(zip(name,count))
count_order=sorted(count_dict.items(),key=lambda x:x[1],reverse=True)
print(count_order)
count_list=[]
for i in count_order:
    count_list.append(i[0])

from transformers import RobertaTokenizer, RobertaForQuestionAnswering
import torch
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForQuestionAnswering.from_pretrained('roberta-base')
tokens=count_list

added_tokens = tokenizer.add_tokens(tokens)
tokenizer.add_special_tokens({'additional_special_tokens':["<code>","</code>","<html>","</html>"]})
model.resize_token_embeddings(len(tokenizer))
tokenizer.save_pretrained('/saved_vocab/10000_new_tokens')
