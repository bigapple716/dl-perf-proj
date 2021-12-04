from datasets import load_dataset

wiki_small = load_dataset("wikitext", "wikitext-2-v1")
wiki_large = load_dataset("wikitext", "wikitext-103-v1")

# print dataset info
print('Dataset Info - wikitext-2')
print(wiki_small)
print('Features:', wiki_small['train'].features)
print('Example:', wiki_small['train'][4])