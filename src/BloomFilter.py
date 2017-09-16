"""
Created on Sep 13, 2017

@author: sarah
"""


import mmh3
from bitarray import bitarray


class BloomFilter(set):
    def __init__(self, size, hash_count):
        super(BloomFilter, self).__init__()
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)
        self.size = size
        self.hash_count = hash_count

    def __len__(self):
        return self.size

    def bit_array(self):
        return self.bit_array

    def add(self, item):
        for seed in range(self.hash_count):
            index = mmh3.hash(item, seed) % self.size
            self.bit_array[index] = 1
        return self

    def lookup(self, string):
        for seed in range(self.hash_count):
            result = mmh3.hash(string, seed) % self.size
            if self.bit_array[result] == 0:
                return False
        return True

    def make_bit_array(self, processed_dictionary):
        for index, word in enumerate(processed_dictionary):
            self.add(word[0])