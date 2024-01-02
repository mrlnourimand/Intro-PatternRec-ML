"""
DATA.ML.100: Introduction to Pattern Recognition and Machine Learning
Ex 03, title: Search analogy.

(This program does a search analogy. For example, "king is to queen as prince is
 to X" - what would be X? In the word embedding space this can be done using the
difference vectors. If x is the king word vector, y is the queen word vector
and z is the prince word vector, then the Euclidean version of analogy is
z = z + (y - x), that is, the vector from king to queen is added to prince to
obtain vector z that is relatively in the same location as queen is from king.
The corresponding word must then be sought using the nearest neighbor search.)
I used np.argsort() function, which computes the indirect sorting of an array
(here is list). It returns a list of indices along the given axis of the same
shape as the input array, in sorted order.

Creator: Maral Nourimand
Student id number: 151749113
Email: maral.nourimand@tuni.fi
"""

import numpy as np


def find_key_by_value(dictionary, target_array):
    for key, value in dictionary.items():
        if len(value) == len(target_array) and all(x == y for x, y in zip(value, target_array)):
            return key
    return None  # Return None if the vector is not found in the dictionary


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
    :return: sorted_list, list of sorted indices from min distance to max distance
    """
    distance_list = []
    for key in vector_dict:
        sub_vector = vector_dict[key] - array
        # calculate the distance of each word from the target vector
        distance_list.append(np.linalg.norm(sub_vector))

    # list of the indices sorted from min distance to max distance
    sorted_list = np.argsort(distance_list)
    # the first 3 value of this list shows the indices of the 3 nearest
    # neighbor words.
    return sorted_list


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

    # Main loop for analogy
    while True:
        input_term = input("\nEnter three words (EXIT to break): ")
        if input_term == 'EXIT':
            break
        else:
            x, y, xp = input_term.rstrip().split()
            # yp = (y - x) + xp
            yp = np.subtract(vectors[y], vectors[x]) + vectors[xp]

            # to check if the exact target vector is located in the txt file.
            result = find_key_by_value(vectors, yp)

            if result:
                print(f"{x} is to {y} as {xp} is to {result}")
            else:
                # print("Array not found in the dictionary.")
                index_list = nearest_neighbor_search(yp, vectors)
                best_matches = []
                for index in index_list[0:5]:
                    # to exclude the 3 input words from the result of the search
                    if ivocab[index] not in [x, y, xp]:
                        best_matches.append(ivocab[index])
                print(f"{x} is to {y} as {xp} is to {best_matches[0]}/"
                      f"{best_matches[1]}")


if __name__ == "__main__":
    main()
