import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from collections import Counter
import string

class ShakespeareDataset(Dataset):
    def __init__(self, file_name):
        data = open(file_name, 'r').read()
        words = self.doc2words(data)
        self.words = self.removepunct(words)
        vocab = self.getvocab(self.words)
        self.vocab_size = len(vocab)
        self.int_to_vocab = {k:w for k,w in enumerate(vocab)}
        self.vocab_to_int = {w:k for k,w in self.int_to_vocab.items()}
        self.seq_len = 32
        print('data has %d words, %d unique.' % (len(words), self.vocab_size))

    def __getitem__(self, index):
        x = [self.vocab_to_int[word] for word in self.words[(index * self.seq_len):(index*self.seq_len) + self.seq_len + 1]]
        x = torch.tensor([x])
        x = x.type(torch.FloatTensor)
        y = torch.zeros_like(x)
        y[:, :-1] = x[:, 1:]
        x = x[:, :-1]
        y = y[:, :-1]
        return x, y

    def __len__(self):
        return len(self.words) // self.seq_len

    def params(self):
        return self.vocab_size, self.int_to_vocab, self.vocab_to_int
      
    def doc2words(self, data):
        lines = data.split('\n')
        lines = [line.strip(r'\"') for line in lines]
        words = ' '.join(lines).split()
        return words
    def removepunct(self, words):
        punct = set(string.punctuation)
        words = [''.join([char for char in list(word) if char not in punct]) for word in words]
        return words
    

    # get vocab from word list
    def getvocab(self, words):
        wordfreq = Counter(words)
        sorted_wordfreq = sorted(wordfreq, key=wordfreq.get)
        return sorted_wordfreq
    
