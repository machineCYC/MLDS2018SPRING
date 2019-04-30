import os
import json
import argparse
import numpy as np
import re

from src.constants import *
from collections import Counter


def build_counter(jason_fullpath):
    """
    """
    video_caption = json.load(open(jason_fullpath, "r"))

    word_counter = Counter()
    for video in video_caption:
        caption = video["caption"]
        for sentence in caption:
            sentence = clean_sentence(sentence)
            words = sentence.split()
            word_counter.update(words)
    return word_counter

def clean_sentence(sentence):
    """
    """
    sentence = sentence.lower()
    sentence = re.sub(r"\b", " ", sentence) # 單詞邊界都加上空格
    sentence = re.sub("[^a-z0-9 ,.]", "", sentence) # 除了a-z 和 0-9 和 , 和 . 和 "空格" 其他都替換成 ""
    return sentence

def clip_word_frequence(num, counter):
    """
    Remove word with frequence less than num.
    
    num: int, at leat number size
    counter: 
    """
    cut_index = None
    sorted_counter = counter.most_common()
    for idx, item in enumerate(sorted_counter):
        if item[1] < num:
            cut_index = idx
            break
    sorted_counter = counter.most_common(cut_index)
    return sorted_counter

def build_dictionary(counter, dictionary_full_path=None):
    """
    Bulid a dictionary with index, words and frequence.
    """
    dic = {}
    for idx, item in enumerate(counter):
        dic[item[0]] = idx

    if dictionary_full_path is not None:
        with open(dictionary_full_path, "w") as f:
            for idx, item in enumerate(counter):
                f.write(str(idx) + " " + item[0] + " " + str(item[1]) + "\n")
    return dic

def transfer_word2index(word2index, label_fullpath, save_label_fullpath):
    """
    """
    max_sentence_length = 0

    with open(label_fullpath, "r") as f:
        label = json.load(f)
    
    transfer_label = []
    for data in label:
        transfer_data = {}
        transfer_data["id"] = data["id"]
        transfer_data["caption"] = []
        for sentence in data["caption"]:
            sentence = clean_sentence(sentence)
            words = sentence.split()

            if len(words) >= max_sentence_length:
                max_sentence_length = len(words)
            
            sentence_index = []
            for word in words:
                if word in word2index:
                    sentence_index.append(word2index[word])
                else:
                    sentence_index.append(word2index[UNK_INDEX])
            transfer_data["caption"].append(sentence_index)
        transfer_label.append(transfer_data)
    
    if save_label_fullpath is not None:
        with open(save_label_fullpath, "w") as f:
            json.dump(transfer_label, f, sort_keys=True, indent=4)
    
    print("max sentence length:{}".format(max_sentence_length))
    return transfer_label

def padding_sentence(word2index, transfer_label_fullpath, transfer_save_label_fullpath, sentence_len):
    """
    """
    with open(transfer_label_fullpath, "r") as f:
        transfer_label = json.load(f)
    
    padding_transfer_label = []
    for data in transfer_label:
        padding_data = {}
        padding_data["id"] = data["id"]
        padding_data["caption"] = []
        for sentence in data["caption"]:
            padding_sentence = [word2index[EOS_INDEX] for _ in range(sentence_len)]
            if len(sentence) >= sentence_len:
                padding_sentence = sentence[:sentence_len]
            else:
                padding_sentence[:len(sentence)] = sentence
            
            padding_sentence += [word2index[EOS_INDEX]]
            padding_sentence = [word2index[BOS_INDEX]] + padding_sentence
            # padding_sentence = sentence
            # padding_sentence += [word2index[EOS_INDEX]]
            # padding_sentence = [word2index[BOS_INDEX]] + padding_sentence
            
            padding_data["caption"].append(padding_sentence)
        padding_transfer_label.append(padding_data)
    
    if transfer_save_label_fullpath is not None:
        with open(transfer_save_label_fullpath, "w") as f:
            json.dump(padding_transfer_label, f, sort_keys=True, indent=4)
    
    return padding_transfer_label


def main(args):
    # build counter
    word_counter = build_counter(jason_fullpath=args.WORD_COUNTER_JSON_PATH)
     
    word_counter = clip_word_frequence(num=3, counter=word_counter) # num=3, 2889 words num=0, 5941
    word_counter.extend([("<EOS>", 0), ("<BOS>", 0), ("<UNK>", 0)])
    
    # build dictionary
    dictionary_full_path = None
    # dictionary_full_path = os.path.join(os.path.dirname(args.WORD_COUNTER_JSON_PATH), "dictionary.txt")
    word2index = build_dictionary(counter=word_counter, dictionary_full_path=dictionary_full_path)
    
    # Transfer training caption to index (word -- > index)
    transfer_label = transfer_word2index(
        word2index=word2index, label_fullpath=args.LABEL_JASON_PATH,
        save_label_fullpath=args.TRANSFER_LABEL_JASON_PATH)
    
    # padding transfer caption with same length
    padding_transfer_label = padding_sentence(
        word2index=word2index, transfer_label_fullpath=args.TRANSFER_LABEL_JASON_PATH,
        transfer_save_label_fullpath=args.PADDING_LABEL_JASON_PATH, sentence_len=46)

if __name__ == "__main__":
    PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "data/MLDS_hw2_1_data")
    WORD_COUNTER_JSON_PATH = os.path.join(DATA_DIR_PATH, "training_label.json")
    
    LABEL_JASON_PATH = os.path.join(DATA_DIR_PATH, "testing_label.json")
    TRANSFER_LABEL_JASON_PATH = LABEL_JASON_PATH.replace("testing_label.json", "transfer_testing_label.json")
    PADDING_LABEL_JASON_PATH = LABEL_JASON_PATH.replace("testing_label.json", "padding_testing_label.json")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--DATA_DIR_PATH",
        type=str,
        default=DATA_DIR_PATH,
        help=""
    )
    parser.add_argument(
        "--WORD_COUNTER_JSON_PATH",
        type=str,
        default=WORD_COUNTER_JSON_PATH,
        help=""
    )
    parser.add_argument(
        "--LABEL_JASON_PATH",
        type=str,
        default=LABEL_JASON_PATH,
        help=""
    )
    parser.add_argument(
        "--TRANSFER_LABEL_JASON_PATH",
        type=str,
        default=TRANSFER_LABEL_JASON_PATH,
        help=""
    )
    parser.add_argument(
        "--PADDING_LABEL_JASON_PATH",
        type=str,
        default=PADDING_LABEL_JASON_PATH,
        help=""
    )
    main(parser.parse_args())
