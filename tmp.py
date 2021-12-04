import time

from transformers import DistilBertTokenizer, TFDistilBertModel
from transformers import pipeline

text = "I love deep learning."
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
encoded_input = tokenizer(text, return_tensors='tf')

start_time = time.time()
model(encoded_input)
print(time.time() - start_time)

# 完形填空, fill mask, masked language modeling
unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
print(unmasker("Hello I'm a [MASK] model.")[0])

# pre-training BERT from scratch
from transformers import BertTokenizer, TFBertForMaskedLM
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForMaskedLM.from_pretrained('bert-base-uncased')

inputs = tokenizer("The capital of France is [MASK].", return_tensors="tf")
inputs["labels"] = tokenizer("The capital of France is Paris.", return_tensors="tf")["input_ids"]

outputs = model(inputs)
loss = outputs.loss
logits = outputs.logits