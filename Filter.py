from src import BloomFilter
from src import SpamFilter
import numpy as np
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB


def main():
    train_dir = 'train-mails'
    test_dir = 'test-mails'
    bloom = BloomFilter.BloomFilter(3000000, 10)
    spam = SpamFilter.SpamFilter(train_dir, test_dir, train_dir, bloom,702)

    dictionary = spam.make_dictionary()
    bloom.make_bit_array(dictionary)
    dictionary = list([i[0] for i in dictionary])

    spam.mail_dir = train_dir
    train_matrix = spam.extract_features(dictionary)

    support_vector_machine_model = LinearSVC()
    support_vector_machine_model.fit(train_matrix, spam.labels)

    spam.mail_dir = test_dir
    test_matrix = spam.extract_features(dictionary)

    #test_labels = np.zeros(260)
    #test_labels[131:260] = 1

    test_labels = spam.read_test()
    support_vector_machine_result = support_vector_machine_model.predict(test_matrix)
    print(confusion_matrix(test_labels, support_vector_machine_result))


if __name__ == '__main__':
    main()
