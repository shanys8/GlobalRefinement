from mpl_toolkits import mplot3d
import numpy as np
from scipy.linalg import expm, logm, sqrtm, fractional_matrix_power
import matplotlib.pyplot as plt
from numpy.linalg import inv, matrix_power

#######################################################################################################################
# Running test data - sampled from continuous function a subdivision using Bspline with m=3
# 3 iterations with input SPD data and avg func of SO mean
#######################################################################################################################


def main():
    d = 3  # dimension of matrix input data
    iterations = 4  # num of iterations for applying scheme

    # prepare test SPD(d) data
    x = np.arange(-5, 6)
    vals = np.zeros((np.size(x), d, d))
    mat_fixed = np.random.rand(d, d)
    # test
    mat_fixed = np.array([[0.5502,    0.2077,    0.2305], [0.6225,    0.3012,    0.8443], [0.5870,   0.4709,    0.1948]])

    mat_fixed = mat_fixed + mat_fixed.T
    for j in range(np.size(x)):
        vals[j, :, :] = expm(abs(x[j]) * (np.dot(mat_fixed, mat_fixed) - mat_fixed))

    # apply subdivision scheme and get refined values
    new_x, refined_vals = subdivision_schema(x, vals, iterations)

    plot_data(new_x, refined_vals)

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


def plot_data(new_x, refined_vals):

    trace_list = list(map(lambda x: np.trace(x), refined_vals))

    plt.scatter(new_x, trace_list)
    plt.show()


    return


if __name__ == "__main__":
    main()
