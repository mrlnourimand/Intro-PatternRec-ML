"""
DATA.ML.100: Introduction to Pattern Recognition and Machine Learning
Ex 02, title: Linear model fit with N > 2 training points.

(This Python program asks user to give N points with a mouse (left click: add
point, right click: stop collecting) and then plots the points and a Fitted
linear model. The function my_linfit(x,y) solves and returns a and b. I used my
own derivations in the function using LSM.)

Creator: Maral Nourimand
"""

import matplotlib.pyplot as plt
import numpy as np


# Linear solver
def my_linfit(x, y):
    """
    This function is to find the parameters of the fitted line, when we use
    linear model for our training points(N>2). It receives the x,y values of the
    points and finds a and b of the fitted line in linear model, using LSM
    (least square model).

    :param x: list, list of float numbers. It's x value of the collected points
    (= training points).
    :param y: list, list of float numbers. It's y value of the collected points
    (= training points).
    :return: float: a and b, two float parameters of the fitted line.
    """
    s_xy = 0
    s_xx = 0
    xmean = sum(x)/len(x)
    ymean = sum(y)/len(y)
    for i in range(0,len(x)):
        s_xy += (x[i] - xmean) * (y[i] - ymean)
        s_xx += (x[i] - xmean) ** 2

    a = s_xy / s_xx
    b = ymean - a * xmean
    return a, b


def interactive_plot():
    """
    This function plots an empty 2D Figure, asks user to give N points with
    a mouse (left click: add point, right click: stop collecting). It shows the
    collected points on the Figure by a blue small circle.

    :return: list, 2 lists of x and y from the collected points
    """
    # Create a plot with axes
    fig, ax = plt.subplots()
    ax.set_xlim(0, 5)
    ax.set_ylim(0,5)

    # Initialize lists to store the coordinates
    x_pnts = []
    y_pnts = []

    # Function to handle mouse click events
    def onclick(event):
        if event.button == 1:  # Left mouse button
            x_pnts.append(event.xdata)
            y_pnts.append(event.ydata)
            ax.scatter(x_pnts, y_pnts, c='b', marker='o', label='Points')

            plt.draw()
        elif event.button == 3:  # Right mouse button
            plt.close()

    # Connect the mouse click event to the handler
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Set axis labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')

    # Set plot title
    plt.title('Interactive 2D Point Plotter')

    # Show the plot
    plt.show()
    return x_pnts, y_pnts


def main():

    # call the function to collect the points from the user
    x_points, y_points = interactive_plot()

    # Display the collected points
    if len(x_points) > 0:
        print("Collected Points:")
        for i, (x, y) in enumerate(zip(x_points, y_points), 1):
            print(f"Point {i}: ({x:.4f}, {y:.4f})")
    else:
        print("No points were collected.")

    # if the points were randomly assigned, instead of asking from the user
    # x = np.random.uniform(-2, 5, 10)
    # y = np.random.uniform(0, 3, 10)
    # plt.plot(x, y, 'kx')
    # xp = np.arange(-2, 5, 0.1)
    # plt.plot(xp, a*xp+b, 'r-')

    # to call the function and find a and b for the fitted line y=ax+b
    a, b = my_linfit(x_points, y_points)

    # to plot the collected points (training points)
    plt.plot(x_points, y_points, 'kx')

    xp = np.arange(min(x_points), max(x_points) + 0.1, 0.1)
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.plot(xp, a * xp + b, 'r-')  # to draw the fitted line with estimated y
    print(f"My fit: a={a:.4f} and b={b:.4f}")

    # Set plot title
    plt.title('Fitted Line Model')
    plt.show()


if __name__ == "__main__":
    main()
