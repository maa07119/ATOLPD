import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from collections import Counter
import string
import os
import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg
import random
import json


class TextDataset(Dataset):
    def __init__(self, data, vocab, seq_length=128, is_train=False):
        # vocab = set(string.ascii_lowercase + string.ascii_uppercase + ' ' + '\n')
        self.data = self.removepunct(data, vocab)
        self.vocab_size = len(vocab)
        self.int_to_vocab = {k:w for k,w in enumerate(vocab)}
        self.vocab_to_int = {w:k for k,w in self.int_to_vocab.items()}
        self.seq_len = seq_length
        self.is_train = is_train
    
    def __getitem__(self, index):
        if self.is_train:
            index = torch.randint(len(self.data) - self.seq_len, (1,))
        
        x = [self.vocab_to_int[char] for char in self.data[index: index + self.seq_len]]
        y = [self.vocab_to_int[char] for char in self.data[index + 1: index + self.seq_len + 1]]
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

    def __len__(self):
        # Think of having length of data  - seq_len
        if self.is_train:
            return len(self.data) - self.seq_len
        else:
            return len(self.data) // self.seq_len
            
        # return len(self.data) - self.seq_len


    def params(self):
        return self.vocab_size, self.int_to_vocab, self.vocab_to_int

    def removepunct(self, data, vocab):
        # data = data.strip('\n')
        # data = ''.join([char for char in data if char in vocab or char in vocab])
        return data
    



def split_data(data, train_split, seq_length=128):
    vocab = string.ascii_lowercase + string.ascii_uppercase + " " + "\n" + string.punctuation + string.digits
    vocab_size = len(vocab)
    data = clean_text(data, vocab)
    train_data, test_data = data[:int(len(data)*train_split)], data[int(len(data)*train_split):]
    train_data = TextDataset(train_data, vocab, seq_length, is_train=True)
    test_data = TextDataset(test_data, vocab, seq_length, is_train=False)
    return train_data, test_data

def read_files(dir_path):
        files = os.listdir(dir_path)
        data = ''
        for file in files:
            if ".txt" not in file:
                continue
            with open(dir_path + file, 'r') as f:
                data += f.read()
        return data


def check_correct_word(word, vocab):
    for c in word:
        if c not in vocab:
            return False
    return True

def clean_text(text, vocab):
    vocab = set(vocab)
    s = ""
    for word in text.split(" "):
        if check_correct_word(word, vocab):
            s += " " + word
    return s[1:]


def get_dataset(dataset_name, train_split = 0.8, seq_length=128):

    if dataset_name == "shakespeare":
        file_name = 'shakespeare.txt'
        data = open(file_name, 'r').read()
        


    elif dataset_name == "austen":
        austen_files = [item for item in gutenberg.fileids() if 'austen' in item]
        # Select Jane Austen's novels
        data = ""
        for f in austen_files:
            data += gutenberg.raw(f)

    elif dataset_name == "dickens":
        data = read_files('dickens/')

    elif dataset_name == "shakespeare_fl":
        file_path = 'shakespeare_fl.json'

        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Convert list of strings to a single string
        train_text = "".join(item for item in data['train'])
        test_text = "".join(item for item in data['test'])
        data = train_text + test_text

    else:
        raise ValueError("Dataset not found")
    
    train_dataset, test_dataset = split_data(data, train_split, seq_length=seq_length)
    return train_dataset, test_dataset


