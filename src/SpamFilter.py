import os
import numpy as np
from collections import Counter


class SpamFilter(set):
    labels = None

    def __init__(self, train_dir, test_dir, mail_dir, bloom_filter):
        super(SpamFilter, self).__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.mail_dir = mail_dir
        self.bloom_filter = bloom_filter

    def process_email(self, dictionary):
        list_to_remove = list(dictionary)
        for item in list_to_remove:
            # all to lower
            item[0].lower()

            # Strip html tags
            item[0].replace('<[^<>]+>', '')

            # replace all numbers with the word number
            item[0].replace('[0-9]+', 'number')

            # replace all links with the word LinkAddress
            item[0].replace('(http|https)://[^\s]*', 'LinkAddress')

            # replace all links with the word EmailAddress
            item[0].replace('[^\s]+@[^\s]+', 'EmailAddress')

            # Remove punctuation
            item[0].replace('\'|&|#|{|}|;', '')
        return dictionary

    def make_dictionary(self):
        emails = [os.path.join(self.train_dir, f) for f in os.listdir(self.train_dir)]
        all_words = []
        for mail in emails:
            with open(mail) as m:
                for i, line in enumerate(m):
                    if i == 2:  # Body of email is only 3rd line of text file
                        words = line.split()
                        all_words += words
        dictionary = Counter(all_words)
        processed_dictionary = self.process_email(dictionary)
        processed_dictionary = processed_dictionary.most_common(3000)
        return processed_dictionary

    def extract_features(self, dictionary, size):
        self.labels = np.zeros(size)
        files = [os.path.join(self.mail_dir, fi) for fi in os.listdir(self.mail_dir)]
        counter = 0
        features_matrix = np.zeros((len(files), 3000))
        doc_id = 0;
        for fil in files:
            with open(fil) as fi:
                for i, line in enumerate(fi):
                    if i == 2:
                        words = line.split()
                        for word in words:
                            word_id = 0
                            if self.bloom_filter.lookup(word):
                                try:
                                    word_id = dictionary.index(word)
                                    features_matrix[doc_id, word_id] = words.count(word)
                                except ValueError:
                                    features_matrix[doc_id, word_id] = 0
                            else:
                                features_matrix[doc_id, word_id] = 0
                doc_id = doc_id + 1
            if "spms" in fil:
                self.labels[counter] = 1
            counter = counter + 1
        return features_matrix
