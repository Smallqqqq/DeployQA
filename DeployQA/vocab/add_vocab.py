from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm",
exclude=['morphologizer', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
nlp.max_length = 30000000
nlp.add_pipe('sentencizer')
document = open("Vocab_data.json").read()
doc = nlp(document)
sents_list=[]
for sent in doc.sents:
    sents_list.append(sent.text)
print(sents_list)
# Remove stop words and punctuation symbols

spaccy_tokenizer = [
    token.text for token in doc if (
    token.is_stop == False and \
    token.is_punct == False and \
    token.text.strip() != '' 
    token.text.find("\n") == -1 and \
    token.text.find("\\n") == -1 and \
    token.text.find("\\") == -1 and \
    token.text.find(".") == -1
    ) ]

corpus = doc.sents
vectorizer = TfidfVectorizer(use_idf=True,tokenizer=spaccy_tokenizer,analyzer='word')
X = vectorizer.fit_transform(sents_list)
#print(vectorizer.get_feature_names())
#print(X.shape)

# get idf of tokens
idf_name=vectorizer.get_feature_names()
idf = vectorizer.idf_
# get tokens from most frequent in documents to least frequent
idf_sorted_indexes = sorted(range(len(idf)), key=lambda k: idf[k]*-1)  #token and its idf sorted index number
idf_sorted = idf[idf_sorted_indexes] #token and its index number's idf value
tokens_by_df = np.array(vectorizer.get_feature_names())[idf_sorted_indexes] #token str

# choose the proportion of new tokens to add in vocabulary
pct = 7.6
index_max = len(np.array(idf_sorted)[np.array(idf_sorted)>=pct])
#new_tokens = tokens_by_df[:index_max]
new_tokens = tokens_by_df[1572:11400]
print(len(idf_sorted))
print(len(new_tokens))

from transformers import RobertaTokenizer, RobertaForQuestionAnswering
import torch
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForQuestionAnswering.from_pretrained('roberta-base')
tokens=[]
for token in new_tokens:
    tokens.append(token)

added_tokens = tokenizer.add_tokens(tokens)
model.resize_token_embeddings(len(tokenizer))
tokenizer.save_pretrained('/saved_vocab')


