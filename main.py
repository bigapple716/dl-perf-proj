from transformers import DistilBertTokenizer, TFDistilBertModel

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
text = "I love deep learning."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)

print(output)
