# A Robust Initialization Method for Knowledge Distillation

Deep Learning Performance Final Project

Co-author: Minghui Zhang (mz2824), Zhe Wang (zw2695)

## 1 A Description of the Project and This Repo


## 2 Dataset

- Dataset name: WikiText-2
- Link: https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/
- Citation: 
@misc{merity2016pointer,
      title={Pointer Sentinel Mixture Models},
      author={Stephen Merity and Caiming Xiong and James Bradbury and Richard Socher},
      year={2016},
      eprint={1609.07843},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
- Stats:
  - Data split: 
    - Training set size: 36718
    - Validation set size: 3760
    - Test set size: 4358
  - Size: 13 MB
  - Origin: Raw text from Wikipedia, collected by Salesforce Research

## 3 Frameworks
### Hugging Face
- Transformers Doc: https://huggingface.co/docs/transformers/index
- Hugging Face supports PyTorch & Tensorflow. We use the PyTorch version for this project.

#### Model Hub
- All models: https://huggingface.co/models
- distilbert: https://huggingface.co/distilbert-base-uncased

#### Dataset Hub
- All datasets: https://huggingface.co/datasets
- wikitext: https://huggingface.co/datasets/wikitext

## 4 Tokenizers
We don't re-train tokenizers. Instead, we use the one that comes with BERT and DistilBERT.

## 5 Pre-training the teacher
- For pre-training, we don't use Next Sentence Prediction, just Masked Language Model.

Run the commands below to pre-train:
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

## 6 Train the student with Knowledge Distillation

Run the commands below to preprocess data and distil:
```shell
cd ./distillation
python scripts/binarized_data.py --file_path data/dump.txt --tokenizer_type bert --tokenizer_name bert-base-uncased --dump_file data/binarized_text

python scripts/token_counts.py --data_file data/binarized_text.bert-base-uncased.pickle --token_counts_dump data/token_counts.bert-base-uncased.pickle

python scripts/extract_distilbert.py \
    --model_name ../Models/5/ \
    --dump_checkpoint ../Models/student_5.pth

python train.py \
    --student_type distilbert \
    --student_config training_configs/distilbert-mini-uncased.json \
    --student_pretrained_weights ../Models/student_5.pth \
    --teacher_type bert \
    --teacher_name bert-base-uncased \
    --teacher_loc ../Models/5/ \
    --alpha_ce 5.0 --alpha_mlm 2.0 --alpha_cos 1.0 --alpha_clm 0.0 --mlm \
    --freeze_pos_embs \
    --dump_path ../Models/student_5_after_distil/ \
    --data_file data/binarized_text.bert-base-uncased.pickle \
    --token_counts data/token_counts.bert-base-uncased.pickle \
    --force --n_epoch 1 --batch_size 32 --n_gpu 0
```

## 7 Results and Observations  
### Results
| Role    | Layers | Layer Drop Rate | Train Loss | Eval Loss | Last Loss | Avg Cum Loss |
|:--------|:-------|:----------------|:-----------|:----------|:----------|:-------------|
| Teacher | 4      | 0               | 6.6797     | 6.5397    | -         | -            |
| Teacher | 4      | 0.1             | 6.7317     | 6.5763    | -         | -            |
| Student | 2      | -               | -          | -         | 19.95     | 20.06        |
| Student | 2      | -               | -          | -         | 19.77     | 19.88        |

### Observations
- The intermediate result (teacher's loss) is slightly worse, but the corresponding student can achieve a better result.
- Our initialization method can help the student converge faster than traditional initialization methods.
