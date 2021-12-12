# A Robust Initialization Method for Knowledge Distillation

Deep Learning Performance Final Project

Co-author: Minghui Zhang (mz2824), Zhe Wang (zw2695)

## TO DO
- [ ] How to evaluate models?

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

## Tokenizers
We don't re-train tokenizers. Instead, we use the one that comes with BERT and DistilBERT.

## Pre-training the teacher
- For pre-training, we don't use Next Sentence Prediction, just Masked Language Model.

```shell
python run_mlm_pt.py \
    --tokenizer_name bert-base-uncased \
    --model_type bert \
    --output_dir output \
    --overwrite_output_dir \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 1 
```

## Distil the student

```shell
cd ./distillation
python scripts/binarized_data.py --file_path data/dump.txt --tokenizer_type bert --tokenizer_name bert-base-uncased --dump_file data/binarized_text

python scripts/token_counts.py --data_file data/binarized_text.bert-base-uncased.pickle --token_counts_dump data/token_counts.bert-base-uncased.pickle

python train.py \
    --student_type distilbert \
    --student_config training_configs/distilbert-mini-uncased.json \
    --teacher_type bert \
    --teacher_name bert-base-uncased \
    --alpha_ce 5.0 --alpha_mlm 2.0 --alpha_cos 1.0 --alpha_clm 0.0 --mlm \
    --freeze_pos_embs \
    --dump_path serialization_dir/my_first_training \
    --data_file data/binarized_text.bert-base-uncased.pickle \
    --token_counts data/token_counts.bert-base-uncased.pickle \
    --force --n_epoch 1
```