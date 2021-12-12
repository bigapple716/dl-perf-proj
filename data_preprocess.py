from datasets import load_dataset


def preprocess(use_small_dataset=True):
    if use_small_dataset:
        raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
    else:
        raw_datasets = load_dataset("wikitext", "wikitext-103-raw-v1")

    # print dataset info
    print('Dataset Info:')
    print(raw_datasets)
    print('Features:', raw_datasets['train'].features)
    print('Example:', raw_datasets['train'][4])
    print()

    with open('distillation/data/dump.txt', 'w') as f_out:
        for row in raw_datasets['train']:
            f_out.write(row['text'])


if __name__ == '__main__':
    preprocess()
