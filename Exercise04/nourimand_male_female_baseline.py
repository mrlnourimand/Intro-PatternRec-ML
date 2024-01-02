"""
DATA.ML.100: Introduction to Pattern Recognition and Machine Learning
Ex 04, title: Male and female -Bayesian classifier part (b), Baseline classifier

(This program reads training data and test data from text files,then makes two
classifiers, one assigns a random class to each test sample. The other assigns
the most likely class (highest a priori) to all test samples.
After that it computes their classification accuracy.)

Creator: Maral Nourimand
Student id number: 151749113
Email: maral.nourimand@tuni.fi
"""
import numpy as np


def main():
    # Load the data1(height, weight) and data2(class) from the text file
    data_training1 = np.loadtxt('male_female_X_train.txt', delimiter=' ')
    data_training2 = np.loadtxt('male_female_y_train.txt', delimiter=' ')
    data_test1 = np.loadtxt('male_female_X_test.txt', delimiter=' ')
    data_test2 = np.loadtxt('male_female_y_test.txt', delimiter=' ')

    # Separate the data into height, weight, and class label columns
    heights = data_training1[:, 0]
    # weights = data_training1[:,1]  # no need for this in section (b)
    class_labels = data_training2[:]
    test_heights = data_test1[:, 0]
    # test_weights = data_test1[:, 1]  # no need for this in section(b)
    test_class_labels = data_test2[:]

    # Separate data into male and female based on class labels
    male_heights = heights[class_labels == 0]
    female_heights = heights[class_labels == 1]
    # male_weights = weights[class_labels == 0]  # no need for this in baseline
    # female_weights = weights[class_labels == 1] # no need for this in baseline

    ############################################################################
    # Randomly assign class labels(0 for male, 1 for female) to the test samples
    random_predictions = np.random.randint(0, 2, size=len(test_heights))

    # Calculate accuracy by comparing random predictions to true class labels
    accuracy_random = np.mean(random_predictions == test_class_labels)
    print(f"Random Classifier Accuracy: {accuracy_random * 100:.4f}%")
    ############################################################################

    ############################################################################
    # Calculate the prior probabilities based on the training data
    prior_male = len(male_heights) / (len(male_heights) + len(female_heights))
    prior_female = len(female_heights) / (
                len(male_heights) + len(female_heights))
    # print(prior_male,prior_female)

    # Decide the majority class based on the priors. Here is Female.
    majority_class = 0 if prior_male > prior_female else 1

    # Assign the majority class label to all test samples with full_like()
    # full_like() function returns an entire array with the same shape and type
    # as the array that was passed to it
    majority_predictions = np.full_like(test_heights, majority_class)

    # Calculate accuracy by comparing majority predictions to true class labels
    accuracy_majority = np.mean(majority_predictions == test_class_labels)
    print(f"Majority Classifier Accuracy: {accuracy_majority * 100:.4f}%")
    ############################################################################


if __name__ == "__main__":
    main()
