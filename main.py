import numpy as np
from scipy.linalg import expm


def main():

    d = 2  # dimension of matrix input data
    iterations = 3  # num of iterations for applying scheme

    # prepare test data
    x = np.arange(-3, 4)
    vals = np.zeros((np.size(x), d, d))

    mat_fixed = np.random.rand(d, d)
    mat_fixed = mat_fixed - mat_fixed.T

    for j in range(np.size(x)):
        mu = expm((x[j] / 2) * mat_fixed)
        mu = mu * mu
        vals[j, :, :] = mu

    # apply subdivision scheme and get refined values
    new_x, refined_vals = subdivision_schema(x, vals, iterations)

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
    # refined_vals[0, :, :] = bspline_rules(vals[0:3, :, :], is_even_indices=True)
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


# spline scheme from lecture 6
def bspline_rules(vals, is_even_indices):
    if is_even_indices:
        one_avg = avg_func(vals[0, :, :], vals[2, :, :], 0.5)
        new_val = avg_func(vals[1, :, :], one_avg, 0.25)
    else:
        new_val = avg_func(vals[0, :, :], vals[1, :, :], 0.5)
    return new_val


def avg_func(a, b, w):
    return (1 - w) * a + w * b


if __name__ == "__main__":
    main()
