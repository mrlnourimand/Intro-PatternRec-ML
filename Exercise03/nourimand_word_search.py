"""
DATA.ML.100: Introduction to Pattern Recognition and Machine Learning
Ex 03, title: Search similar words.

(This program read a txt file of 400,000 words with their 50 dim vectors in
space for any input word, then for any input word returns the three (3) most
similar words (the most similar should be the input word itself)).

Creator: Maral Nourimand
"""

import numpy as np


def nearest_neighbor_search(array, vector_dict):
    """
    This function receives the dictionary of the words and their
    vectors(vector_dict), and the target vector(array). Then it calculates the
    distance of each word from that vector. It sorts the distances from min to
    max, then returns the list of indices of the sorted words. For example if
    sorted_list[0] = 265, it means that the 265th word is the closest neighbor
    to the given array(vector)

    :param array: list, vector of 50 dim
    :param vector_dict: dictionary, (key=word:value=50 dim. vector of each word)
    :return: sorted_list and distance_list, list of sorted indices from min
            distance to max distance. And a list of each word's distance from
            that particular array.
    """
    distance_list = []
    for key in vector_dict:
        sub_vector = np.subtract(vector_dict[key], array)
        # to calculate the distance of each word from the target vector
        distance_list.append(np.linalg.norm(sub_vector))

    # list of the indices sorted from min distance to max distance
    # the first 3 value of this list shows the indices of the 3 nearest
    # neighbor words.
    sorted_list = np.argsort(distance_list)

    return sorted_list, distance_list


def main():
    vocabulary_file = 'word_embeddings.txt'

    # Read words
    print('Read words...')
    with open(vocabulary_file, 'r', encoding="utf8") as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]

    # Read word vectors. dictionary vectors{key=word:value=50-dimension vector}
    print('Read word vectors...')
    with open(vocabulary_file, 'r', encoding="utf8") as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    # to convert the words(list) into vocab(dict{word:index}) and
    # ivocab(dict{index:word})
    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}
    # print("vocab: ", type(vocab), "ivocab: ", type(ivocab), vocab["."])

    # Vocabulary and inverse vocabulary (dict objects)
    print('Vocabulary size')
    print(len(vocab))  # size of the vocabulary dictionary = 400,000.
    print(vocab['man'])  # "man" in the text file is the 300th word.
    print(len(ivocab))  # size of the inverse vocabulary dictionary = 400,000.
    print(ivocab[10])  # print the 10th word in the file = "for"

    # W contains vectors for
    # print('Vocabulary word vectors')
    # vector_dim = len(vectors[ivocab[0]])
    # W = np.zeros((vocab_size, vector_dim))
    # for word, v in vectors.items():
    #     if word == '<unk>':
    #         continue
    #     W[vocab[word], :] = v
    # print(W.shape)

    while True:
        input_word = input("\nEnter a word to find its "
                           "neighbors (EXIT to break): ")
        if input_word == 'EXIT':
            break
        else:
            index_list, distance = nearest_neighbor_search(vectors[input_word],
                                                           vectors)
            print("\n                             Word       Distance")
            print("---------------------------------------------------------")
            print("%33s\t\t%f\n" % (ivocab[index_list[0]],
                                    distance[index_list[0]]))

            print("%33s\t\t%f\n" % (ivocab[index_list[1]],
                                    distance[index_list[1]]))

            print("%33s\t\t%f\n" % (ivocab[index_list[2]],
                                    distance[index_list[2]]))


if __name__ == "__main__":
    main()
