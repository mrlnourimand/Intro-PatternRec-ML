"""
DATA.ML.100: Introduction to Pattern Recognition and Machine Learning
Ex 04, title: Male and female - Bayesian classifier part (c):Bayes classifier
                                with non-parametric distribution.

(This program reads training data and test data from text files, then makes
classifier based on Bayesian theory. For example;
p(male|height) = p(height|male)*p(male)/p(height). If p(male|height)>p(female|height),
it will assign class MALE to the test item.
It does the classification based on heights only, wights only and
the weight and height together(multiply likelihoods).
Next, it computes the accuracy of the 3 aforementioned classification algorithm
and prints them.

Creator: Maral Nourimand
"""
import numpy as np


def main():
    # Load the data1(height, weight) and data2(class) from the text files
    data_training1 = np.loadtxt('male_female_X_train.txt', delimiter=' ')
    data2_training2 = np.loadtxt('male_female_y_train.txt', delimiter=' ')
    data_test1 = np.loadtxt('male_female_X_test.txt', delimiter=' ')
    data_test2 = np.loadtxt('male_female_y_test.txt', delimiter=' ')

    # Separate the data into height, weight, and class label columns
    heights = data_training1[:, 0]
    weights = data_training1[:, 1]
    class_labels = data2_training2[:]
    test_heights = data_test1[:, 0]
    test_weights = data_test1[:, 1]
    test_class_labels = data_test2[:]

    # Separate data into male and female based on class labels
    male_heights = heights[class_labels == 0]
    female_heights = heights[class_labels == 1]
    male_weights = weights[class_labels == 0]
    female_weights = weights[class_labels == 1]

    # Step 1: Compute Prior Probabilities
    prior_male = len(male_heights) / (len(male_heights) + len(female_heights))
    prior_female = len(female_heights) / (
                len(male_heights) + len(female_heights))
    print(f"Prior probabilities for male is {prior_male:.4f}")
    print(f"Prior probabilities for female is {prior_female:.4f}")

    # Step 2: Compute Class Likelihoods (Height and Weight and joint)
    num_bins = 10  # Number of bins for histograms

    # Compute histograms for height and weight separately for male and female
    hist_male_heights, bin_edges_heights = np.histogram(male_heights,
                                                        bins=num_bins,
                                                        range=(80, 220))
    hist_female_heights, _ = np.histogram(female_heights, bins=num_bins,
                                          range=(80,220))

    hist_male_weights, bin_edges_weights = np.histogram(male_weights,
                                                        bins=num_bins,
                                                        range=(30, 180))
    hist_female_weights, _ = np.histogram(female_weights, bins=num_bins,
                                          range=(30,180))

    # Compute centroids of bins
    bin_centers_heights = (bin_edges_heights[
                                :-1] + bin_edges_heights[1:]) / 2
    bin_centers_weights = (bin_edges_weights[
                                :-1] + bin_edges_weights[1:]) / 2

    # Initialize arrays to store likelihoods
    likelihood_height_male = np.zeros(len(test_heights))
    likelihood_height_female = np.zeros(len(test_heights))

    likelihood_weight_male = np.zeros(len(test_weights))
    likelihood_weight_female = np.zeros(len(test_weights))

    # Assign each test sample to the closest bin and calculate likelihoods
    for i in range(len(test_heights)):
        # Find the closest bin for height
        bin_index_height = np.argmin(
            np.abs(bin_centers_heights - test_heights[i]))
        # Calculate likelihoods for height. P(height|male) and P(height|female)
        likelihood_height_male[i] = hist_male_heights[bin_index_height] / len(male_heights)
        likelihood_height_female[i] = hist_female_heights[ bin_index_height] / len(female_heights)

    for i in range(len(test_weights)):
        # Find the closest bin for weight
        bin_index_weight = np.argmin(np.abs(bin_centers_weights - test_weights[i]))
        # Calculate likelihoods for weight. P(weight|male), P(weight|female)
        likelihood_weight_male[i] = hist_male_weights[bin_index_weight] / len(male_weights)
        likelihood_weight_female[i] = hist_female_weights[bin_index_weight] / len(female_weights)

    # Combine likelihoods for heights and weights using multiplication
    # (Naive Bayes assumption)
    joint_likelihood_male = likelihood_height_male * likelihood_weight_male
    joint_likelihood_female = likelihood_height_female * likelihood_weight_female

    # Step 3: Classification
    # Classify test samples based on maximum likelihood
    # heights only
    predictions_height = (prior_male * likelihood_height_male) < (
            prior_female * likelihood_height_female)
    # weights only
    predictions_weight = (prior_male * likelihood_weight_male) < (
            prior_female * likelihood_weight_female)
    # heights and weights together(joint likelihood)
    predictions_joint = (prior_male * joint_likelihood_male) < (
            prior_female * joint_likelihood_female)

    # Calculate accuracy for height only, weight only, and  height&weight joint
    accuracy_height_only = np.mean(predictions_height == test_class_labels)
    accuracy_weight_only = np.mean(predictions_weight == test_class_labels)
    accuracy_height_weight_joint = np.mean(predictions_joint == test_class_labels)

    print(f"Accuracy (Height Only): {accuracy_height_only * 100:.4f}%")
    print(f"Accuracy (Weight Only): {accuracy_weight_only * 100:.4f}%")
    print(f"Accuracy (Height and Weight Together): "
          f"{accuracy_height_weight_joint * 100:.4f}%")


if __name__ == "__main__":
    main()
