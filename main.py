from transformers import BertConfig
from transformers import BertTokenizer, TFBertForMaskedLM

# Initializing a BERT bert-base-uncased style configuration
configuration = BertConfig()

# Initializing a model from the bert-base-uncased style configuration
teacher = TFBertForMaskedLM(configuration)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

inputs = tokenizer("The capital of France is [MASK].", return_tensors="tf")
inputs["labels"] = tokenizer("The capital of France is Paris.", return_tensors="tf")["input_ids"]

teacher.compile()
teacher.fit()
