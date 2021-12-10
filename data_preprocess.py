from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

wiki_small = load_dataset("wikitext", "wikitext-2-v1")
wiki_large = load_dataset("wikitext", "wikitext-103-v1")

raw_datasets = wiki_small

# print dataset info
print('Dataset Info - wikitext-2')
print(wiki_small)
print('Features:', wiki_small['train'].features)
print('Example:', wiki_small['train'][4])

batches = tokenizer(raw_datasets, max_length=512, padding='max_length', truncation=True)
print(batches)
