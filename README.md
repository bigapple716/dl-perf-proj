# A Robust Initialization Method for Knowledge Distillation

Deep Learning Performance Final Project

Co-author: Minghui Zhang (mz2824), Zhe Wang (zw2695)

## Dataset Analysis

- Dataset: WikiText-103
- Link: https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-mo
- Statistics:
  - Number of Records: 101,880,768 tokens
  - Data split: 101425671 training tokens, 213886 validation tokens, 241211 test tokens - Size: 181 MB
  - Origin: Raw text from Wikipedia, collected by Salesforce Research

## Frameworks
### Hugging Face
- Transformers Doc: https://huggingface.co/docs/transformers/index
- Hugging Face supports PyTorch & Tensorflow. We use the Tensorflow version, because it runs much faster on Apple M1 Chip.

#### Models
- All models: https://huggingface.co/models
- distilbert: https://huggingface.co/distilbert-base-uncased