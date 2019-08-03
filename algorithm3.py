from mpl_toolkits import mplot3d
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


#######################################################################################################################
# Algorithm 3 with real and complex values
# #######################################################################################################################


def main():
    d = 3  # dimension of matrix input data
    iterations = 1  # num of iterations for applying scheme
    input_type = 'so'

    s = 3
    real_alphas = [1]
    complex_alphas = [1+1j]   # TODO insert right values of roots -  -(root)^-1  ?

    # prepare test data
    x = np.arange(-5, 6)
    vals = np.zeros((np.size(x), d, d))

    # Input vals
    if input_type == 'so':  # SO(d) data
        mat_fixed = np.random.rand(d, d)
        mat_fixed = mat_fixed - mat_fixed.T
        for j in range(np.size(x)):
            mu = expm((x[j] / 2) * mat_fixed)
            vals[j, :, :] = np.dot(mu, mu)

    elif input_type == 'spd':  # SPD(d) data:
        mat_fixed = np.random.rand(d, d)
        mat_fixed = mat_fixed + mat_fixed.T
        for j in range(np.size(x)):
            vals[j, :, :] = expm(abs(x[j]) * (np.dot(mat_fixed, mat_fixed) - mat_fixed))

    # apply subdivision scheme and get refined values
    new_x, refined_vals = subdivision_schema(x, vals, iterations, real_alphas, complex_alphas)

    # TODO perform shift in s of indices

    plot_data(new_x, refined_vals)

    return


def subdivision_schema(x, vals, iterations, real_alphas, complex_alphas):
    current_vals = vals
    current_x = x

    for _ in range(iterations):
        current_x, current_vals = sn_scheme_refinement(current_x, current_vals, real_alphas, complex_alphas)
    return current_x, current_vals


def sn_scheme_refinement(x, vals, real_alphas, complex_alphas):
    new_len = (np.size(x) - 2) * 2 + 1
    refined_x = np.zeros(new_len)
    refined_vals = np.zeros((new_len, np.shape(vals)[1], np.shape(vals)[2]))

    # iterations on the *real* alpha values
    refined_x[0] = (x[0] + x[1])/2
    refined_vals[0, :, :] = sn_scheme_rules(vals[0:, :, :], real_alphas[0], is_even_indices=False)

    for j in range(2, np.size(x)):
        for alpha in real_alphas:
            current_index = 2 * (j - 1)
            # interpolation points
            refined_x[current_index-1] = x[j-1]
            refined_vals[current_index-1, :, :] = sn_scheme_rules(vals[j-1:, :, :], alpha, is_even_indices=True)
            # new refined points
            refined_x[current_index] = (x[j-1] + x[j]) / 2
            refined_vals[current_index, :, :] = sn_scheme_rules(vals[j-1:, :, :], alpha, is_even_indices=False)

    # iterations on the *complex* alpha values
    refined_x[0] = (x[0] + x[1])/2
    refined_vals[0, :, :] = sn_scheme_rules(vals[0:, :, :], complex_alphas[0])

    for j in range(2, np.size(x)):
        for alpha in complex_alphas:
            current_index = 2 * (j - 1)
            # interpolation points
            refined_x[current_index-1] = x[j-1]
            refined_vals[current_index-1, :, :] = complex_sn_scheme_rules(vals[j-1:, :, :], alpha)
            # new refined points
            refined_x[current_index] = (x[j-1] + x[j]) / 2
            refined_vals[current_index, :, :] = complex_sn_scheme_rules(vals[j-1:, :, :], alpha)

    return refined_x, refined_vals


def complex_sn_scheme_rules(vals, curr_alpha):

    w1 = calculate_w1(curr_alpha)
    w2 = calculate_w2(curr_alpha)
    w3 = calculate_w3(curr_alpha)

    new_val = pyramid(vals[0, :, :], vals[1, :, :], vals[2, :, :], w1, w2, w3, curr_alpha)

    return new_val


def calculate_w1(alpha):
    return 1 / (1 + 2 * np.real(alpha) + np.power(np.absolute(alpha), 2))


def calculate_w2(alpha):
    return 2 * np.real(alpha) / (1 + 2 * np.real(alpha) + np.power(np.absolute(alpha), 2))


def calculate_w3(alpha):
    return np.power(np.absolute(alpha), 2) / (1 + 2 * np.real(alpha) + np.power(np.absolute(alpha), 2))


def calculate_r(alpha):
    return 1 / (1 + np.absolute(alpha))


def calculate_t1(w1, r):
    return w1 / r


def calculate_t2(w3, r):
    return 1 - w3 / (1-r)


def pyramid(p1, p2, p3, w1, w2, w3, alpha):
    r = calculate_r(alpha)
    t1 = calculate_t1(w1, r)
    t2 = calculate_t2(w3, r)
    new_val = avg_func(avg_func(p3, p2, t2), avg_func(p2, p1, t1), r)
    return new_val


def sn_scheme_rules(vals, curr_alpha, is_even_indices):
    if is_even_indices:  # even indices
        new_val = vals[0, :, :]
    else:  # odd indices
        new_val = avg_func(vals[0, :, :], vals[1, :, :], curr_alpha)
    return new_val


def avg_func(a, b, alpha):
    return (1 / (1 + alpha)) * a + (alpha / (1 + alpha)) * b


def plot_data(new_x, refined_vals):
    # TODO by the structure of data
    return


if __name__ == "__main__":
    main()
