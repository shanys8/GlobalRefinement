from mpl_toolkits import mplot3d
import numpy as np
import math
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
    d = 2  # dimension of matrix input data
    iterations = 2  # num of iterations for applying scheme

    # prepare test SPD(d) data
    x = np.arange(-5, 6)
    vals = np.zeros((np.size(x), d, d))
    mat_fixed = np.random.rand(d, d)
    mat_fixed = mat_fixed + mat_fixed.T
    for j in range(np.size(x)):
        vals[j, :, :] = expm(abs(x[j]) * (np.dot(mat_fixed, mat_fixed) - mat_fixed))

    # apply subdivision scheme and get refined values
    new_x, refined_vals = subdivision_schema(x, vals, iterations)

    plot_data(refined_vals)

    return


def subdivision_schema(x, vals, iterations):
    current_vals = vals
    current_x = x

    for _ in range(iterations):
        current_x, current_vals = bspline_refinement(current_x, current_vals)
    return current_x, current_vals


def bspline_refinement(x, vals):
    new_len = (np.size(x) - 2) * 2 + 1
    refined_x = np.zeros(new_len)
    refined_vals = np.zeros((new_len, np.shape(vals)[1], np.shape(vals)[2]))

    # first refined point
    refined_x[0] = (x[0] + x[1])/2
    refined_vals[0, :, :] = bspline_rules(vals[0:, :, :], is_even_indices=False)

    for j in range(2, np.size(x)):
        current_index = 2 * (j - 1)
        # interpolation points
        refined_x[current_index-1] = x[j-1]
        refined_vals[current_index-1, :, :] = bspline_rules(vals[j-2:, :, :], is_even_indices=True)
        # new refined points
        refined_x[current_index] = (x[j-1] + x[j]) / 2
        refined_vals[current_index, :, :] = bspline_rules(vals[j-1:, :, :], is_even_indices=False)

    return refined_x, refined_vals


def bspline_rules(vals, is_even_indices):
    if is_even_indices:  # even indices
        one_avg = avg_func(vals[0, :, :], vals[2, :, :], 0.5)
        new_val = avg_func(vals[1, :, :], one_avg, 0.25)
    else:  # odd indices
        new_val = avg_func(vals[0, :, :], vals[1, :, :], 0.5)
    return new_val


# ALM mean
def avg_func(A, B, w):
    A_half = sqrtm(A)
    A_min_half = inv(A_half)
    midM = fractional_matrix_power(np.dot(np.dot(A_min_half, B), A_min_half),  w)
    return np.dot(np.dot(A_half, midM), A_half)


def plot_data(refined_vals):

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

    center_x = []
    center_y = []

    for j in range(0, np.shape(refined_vals)[0]-1):
        sigma = refined_vals[j, :, :]
        # Compute eigenvalues and associated eigenvectors
        vals, vecs = np.linalg.eigh(sigma)

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
        ellipse.set_facecolor(np.random.rand(3))
        brightness = j / (np.shape(refined_vals)[0]-1)
        color = [0, 0, brightness]
        ellipse.set_facecolor(color)

        ellipse.set_alpha(0.2)
        ax.add_artist(ellipse)

    ax.set_xlim(np.maximum(np.amin(center_x), -100) - 10, np.minimum(np.amax(center_x), 100) + 10)
    ax.set_ylim(np.maximum(np.amin(center_y), -100) - 10, np.minimum(np.amax(center_y), 100) + 10)
    plt.show()

    return


if __name__ == "__main__":
    main()
