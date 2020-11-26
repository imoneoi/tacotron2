from text.preprocessors import FullPreprocessor

import numpy as np
import argparse

# process dataset
def write_text_file(path, text):
    f = open(path, "w", encoding="UTF-8")
    f.write(text)
    f.close()

def process_dataset(path, out, val_split=0.025, val_split_seed=6666):
    # Read content
    f = open(path, "r", encoding="UTF-8")
    content = f.readlines()
    content = [line.rstrip("\n") for line in content]
    f.close()

    text_list = []
    pinyin_list = []
    fileid_list = []

    # Split lines
    n = len(content) // 2
    for idx in range(n):
        line1 = content[idx * 2]
        line2 = content[idx * 2 + 1]

        file_id, text = line1.split("\t", 1)
        pinyin = line2.lstrip("\t")

        fileid_list.append(file_id)
        text_list.append(text)
        pinyin_list.append(pinyin)

    # Preprocess
    preprocessor = FullPreprocessor()
    label_list = preprocessor.preprocess(text_list, pinyin_list)

    # Write labels
    full_set = []
    symbol_list = {}
    for file_id, label in zip(fileid_list, label_list):
        # Append fullset
        full_set.append("DUMMY/{}.wav|{}|{}".format(file_id, label, label))

        # Count symbols
        for item in label.split(" "):
            if not item in symbol_list:
                symbol_list[item] = 0

            symbol_list[item] += 1

    # Train-test split
    np.random.seed(val_split_seed)

    train_set = []
    val_set = []
    for item in full_set:
        if np.random.rand() < val_split:
            val_set.append(item)
        else:
            train_set.append(item)

    # Save result
    write_text_file(out + ".train.txt", "\n".join(train_set))
    write_text_file(out + ".val.txt", "\n".join(val_set))

    # Sort symbol list
    symbol_list = {k: v for k, v in sorted(symbol_list.items(), key=lambda item: item[1], reverse=True)}

    return symbol_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/media/one/Data/datasets/BZNSYP/ProsodyLabeling/000001-010000.txt", help="Path to dataset label file")
    parser.add_argument("--output", default="out", type=str, help="Output dataset filelist")

    args = parser.parse_args()
    symbol_list = process_dataset(args.path, args.output)

    print(symbol_list)