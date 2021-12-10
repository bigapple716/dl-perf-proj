from datasets import load_dataset
from transformers import AutoTokenizer
import tensorflow as tf

# hyper parameters
BATCH_SIZE = 8

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize_function(example):
    return tokenizer(example['text'], max_length=512, padding='max_length', truncation=True)


def preprocess(use_small_dataset=True):
    wiki_small = load_dataset("wikitext", "wikitext-2-raw-v1")
    wiki_large = load_dataset("wikitext", "wikitext-103-raw-v1")

    if use_small_dataset:
        raw_datasets = wiki_small
    else:
        raw_datasets = wiki_large

    # print dataset info
    print('Dataset Info:')
    print(wiki_small)
    print('Features:', wiki_small['train'].features)
    print('Example:', wiki_small['train'][4])
    print()

    with open('dump.txt', 'w') as f_out:
        for row in wiki_small['train']:
            f_out.write(row['text'])

    # # map tokenizer to datasets
    # tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    #
    # # print dataset info
    # print('Tokenized Dataset Info:')
    # print(tokenized_datasets)
    #
    # # Arrow datasets -> tf datasets
    # tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    #     columns=["attention_mask", "input_ids", "token_type_ids"],
    #     shuffle=True,
    #     batch_size=BATCH_SIZE,
    # )
    # tf_validation_dataset = tokenized_datasets["validation"].to_tf_dataset(
    #     columns=["attention_mask", "input_ids", "token_type_ids"],
    #     shuffle=False,
    #     batch_size=BATCH_SIZE,
    # )
    # tf_test_dataset = tokenized_datasets["test"].to_tf_dataset(
    #     columns=["attention_mask", "input_ids", "token_type_ids"],
    #     shuffle=False,
    #     batch_size=BATCH_SIZE,
    # )
    #
    # # print dataset info
    # print('tf datasets:')
    # print(tf_train_dataset)
    # print(tf_validation_dataset)
    # print(tf_test_dataset)
    #
    # return tf_train_dataset, tf_validation_dataset, tf_test_dataset


if __name__ == '__main__':
    preprocess()