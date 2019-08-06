from mpl_toolkits import mplot3d
import numpy as np
import math
import sklearn
from sklearn.datasets import make_spd_matrix
from scipy.linalg import expm, logm, sqrtm, fractional_matrix_power
import matplotlib.pyplot as plt
from numpy.linalg import inv, matrix_power
from matplotlib.patches import Ellipse
import colorsys

#######################################################################################################################
# Running test data - sampled from continuous function a subdivision using Bspline with m=3
# with input SPD data and avg func of ALM mean - draw ellipse from each iteration of refinement
#######################################################################################################################


def main():

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

    d = 2  # dimension of matrix input data

    # p1 = np.array([[1, 2], [3, 4]])
    #
    # p1 = make_spd_matrix(2)
    # p2 = np.rot90(p1, k=1, axes=(1,0))

    p1 = np.array([[2.36691593, 0.47362613], [0.47362613, 0.43641888]])
    # p2 = np.rot90(p1, k=1, axes=(1,0))

    rotation_mat = np.array([[0, -1], [1, 0]])

    p2 = np.dot(p1, rotation_mat)

    plot_data(p1, p2)

    return


# ALM mean
def avg_func(A, B, w):
    A_half = sqrtm(A)
    A_min_half = inv(A_half)
    midM = fractional_matrix_power(np.dot(np.dot(A_min_half, B), A_min_half),  w)
    return np.dot(np.dot(A_half, midM), A_half)


def add_ellipse(ax, matrix, color):
    # Compute eigenvalues and associated eigenvectors
    vals, vecs = np.linalg.eigh(matrix)

    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Eigenvalues give length of ellipse along each eigenvector
    w, h = 2 * np.sqrt(vals)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ellipse = Ellipse([vals[0], vals[1]], w, h, theta)
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_facecolor(color)
    ellipse.set_alpha(0.2)
    ax.add_artist(ellipse)

    return vals


def plot_data(p1, p2):

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

    center_x = []
    center_y = []

    # Compute eigenvalues and associated eigenvectors
    vals, vecs = np.linalg.eigh(p1)

    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Eigenvalues give length of ellipse along each eigenvector
    w, h = 2 * np.sqrt(vals)
    center_x = np.append(center_x, vals[0])
    center_y = np.append(center_y, vals[1])
    ax.tick_params(axis='both', which='major', labelsize=20)
    ellipse = Ellipse([vals[0], vals[1]], w, h, theta)
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_facecolor([1, 0, 0])
    ellipse.set_alpha(0.2)
    ax.add_artist(ellipse)

    # calculate p2

    theta_p2 = theta + 90

    p2_x = math.sqrt(1 / (math.pow(np.tan(np.radians(theta_p2)), 2) + 1))
    p2_y = np.tan(np.radians(theta_p2)) * p2_x

    p2_eigenvectors = np.array([[p2_x, p2_y], [p2_y, p2_x]])
    p2_eigenvalues = np.array([[vals[0], 0], [0, vals[1]]])

    p2 = np.dot(np.dot(p2_eigenvectors, p2_eigenvalues), inv(p2_eigenvectors))

    # Compute eigenvalues and associated eigenvectors
    vals, vecs = np.linalg.eigh(p2)

    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Eigenvalues give length of ellipse along each eigenvector
    w, h = 2 * np.sqrt(vals)
    center_x = np.append(center_x, vals[0])
    center_y = np.append(center_y, vals[1])
    ax.tick_params(axis='both', which='major', labelsize=20)
    ellipse = Ellipse([vals[0], vals[1]], w, h, theta)
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_facecolor([0, 1, 0])
    ellipse.set_alpha(0.2)
    ax.add_artist(ellipse)

    # vals = add_ellipse(ax, p1, [1, 0, 0])
    # center_x = np.append(center_x, vals[0])
    # center_y = np.append(center_y, vals[1])
    #
    # vals = add_ellipse(ax, p2, [0, 1, 0])
    # center_x = np.append(center_x, vals[0])
    # center_y = np.append(center_y, vals[1])

    ax.set_xlim(np.maximum(np.amin(center_x), -100) - 2, np.minimum(np.amax(center_x), 100) + 2)
    ax.set_ylim(np.maximum(np.amin(center_y), -100) - 2, np.minimum(np.amax(center_y), 100) + 2)

    plt.show()

    return


if __name__ == "__main__":
    main()
