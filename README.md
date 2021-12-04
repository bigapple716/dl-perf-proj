# A Robust Initialization Method for Knowledge Distillation

Deep Learning Performance Final Project

Co-author: Minghui Zhang (mz2824), Zhe Wang (zw2695)

## TO DO
- [ ] How to evaluate models?
- [ ] 如何把dataset转化成tf.dataset, 送进模型里去训练
- [ ] How to train teacher model?
- [ ] How to distill student model?
- [ ] How to add LayerDrop during teacher's training?

## Dataset Analysis

- Dataset: WikiText-103
- Link: https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/
- Statistics:
  - Number of Records: 101,880,768 tokens
  - Data split: 101425671 training tokens, 213886 validation tokens, 241211 test tokens - Size: 181 MB
  - Origin: Raw text from Wikipedia, collected by Salesforce Research

## Frameworks
### Hugging Face
- Transformers Doc: https://huggingface.co/docs/transformers/index
- Hugging Face supports PyTorch & Tensorflow. We use the Tensorflow version, because it runs much faster on Apple M1 Chip.

#### Model Hub
- All models: https://huggingface.co/models
- distilbert: https://huggingface.co/distilbert-base-uncased

#### Dataset Hub
- All datasets: https://huggingface.co/datasets
- wikitext: https://huggingface.co/datasets/wikitext

### Tokenizers
We don't re-train tokenizers. Instead, we use the one that comes with BERT and DistilBERT.

### Pre-training the teacher
- For pre-training, we don't use Next Sentence Prediction, just Masked Language Model.
