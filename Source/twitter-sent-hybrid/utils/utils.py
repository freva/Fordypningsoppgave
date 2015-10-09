import numpy as np

"""
    All methods below assume that normalize_test_set_classification_scheme has been called prior
"""
def normalize_test_set_classification_scheme(dataset):
    for i in dataset:
        i[2] = i[2].replace('"', '')
        if i[2] == 'objective' or i[2] == 'objective-OR-neutral':
            i[2] = 'neutral'
    return dataset[:,[3,2]]


def generate_polarity_set(dataset):
    new_set = []
    for i in dataset:
        if i[1] != 'neutral':
            new_set.append(i)
    return np.array(new_set)


def generate_subjective_set(dataset):
    new_set = np.empty_like(dataset)
    new_set[:] = dataset

    for i in new_set:
        if i[1] != 'neutral':
            i[1] = 'subjective'
    return new_set


def generate_two_part_dataset(train):
    subjectivity = generate_subjective_set(train)
    polarity = generate_polarity_set(train)

    return subjectivity, polarity


def reduce_dataset(dataset, num):
    num_n, num_o, num_p = 0, 0, 0
    new_dataset = []
    for i in dataset:
        if i[1] == 'negative' and num_n < num:
            num_n += 1
            new_dataset.append(i)
        if i[1] == 'positive' and num_p < num:
            num_p += 1
            new_dataset.append(i)
        if i[1] == 'neutral' and num_o < num:
            num_o += 1
            new_dataset.append(i)
    return np.array(new_dataset)
