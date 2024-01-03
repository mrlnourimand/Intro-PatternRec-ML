"""
DATA.ML.100: Introduction to Pattern Recognition and Machine Learning
Ex 04, title: Male and female - Bayesian classifier part (a):Height and weight
                                                            histograms.

(This program reads training data from text files, computes the histograms of
 the male height, female height, male weight and female height measurements
  using the NumPy histogram() function. For the both it uses 10 bins and fixed
  ranges ([80, 220] for height and [30, 180] for weight)

Creator: Maral Nourimand
"""

import numpy as np
import matplotlib.pyplot as plt


def main():

    # Load the training data(height, weight) and data2(class) from the text file
    data_training1 = np.loadtxt('male_female_X_train.txt',delimiter=' ')
    data_training2 = np.loadtxt('male_female_y_train.txt', delimiter=' ')

    # Separate the training data into height, weight, and class label columns
    heights = data_training1[:, 0]
    weights = data_training1[:, 1]
    class_labels = data_training2[:]

    # Separate training data into male and female based on class labels
    male_heights = heights[class_labels == 0]
    female_heights = heights[class_labels == 1]
    male_weights = weights[class_labels == 0]
    female_weights = weights[class_labels == 1]

    # histogram computation for male data
    hist_male_heights, bins_height = np.histogram(male_heights, bins=10,
                                                  range=[80,220])
    hist_male_weights, bins_weight = np.histogram(male_weights, bins=10,
                                                  range=[30,180])

    # histogram computation for female data
    hist_female_heights, _ = np.histogram(female_heights, bins=10,
                                          range=[80,220])
    hist_female_weights, _ = np.histogram(female_weights, bins=10,
                                          range=[30,180])

    # Create a histogram for male & female heights
    plt.hist(bins_height[:-1], bins_height, weights=hist_male_heights,
             label='Male', alpha=.5)
    plt.hist(bins_height[:-1],bins_height, weights=hist_female_heights,
             label='Female', alpha=.5)

    # add legend and axis labels
    plt.ylabel('Count')
    plt.xlabel('Height')
    plt.legend(loc='upper right')

    # show labels
    plt.show()

    # Create a histogram for male & female weights
    plt.hist(bins_weight[:-1], bins_weight, weights=hist_male_weights,
             label='Male', alpha=.5)
    plt.hist(bins_weight[:-1], bins_weight, weights=hist_female_weights,
             label='Female', alpha=.5)

    # add legend and axis labels
    plt.ylabel('Count')
    plt.xlabel('Weight')
    plt.legend(loc='upper right')

    # show labels
    plt.show()


if __name__ == "__main__":
    main()
