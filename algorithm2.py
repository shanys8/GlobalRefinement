from mpl_toolkits import mplot3d
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


#######################################################################################################################
# for Sn with n=1:
# symbol is:  a(z) = z^-1 * (1/2 + z + (1/2)z^2)
# roots are: z = -1
# s = 1

# for Sn with n=2:
# symbol is:  a(z) = z^-3 * (1/4 + (1/3)z + (1/4)z^2+ (1/3)z^3 + (1/4)z^4+ (1/3)z^5 + (1/4)z^6)
# roots are: z = -1 and rest of the roots are complex
# s = 3
# #######################################################################################################################


def main():
    d = 3  # dimension of matrix input data
    iterations = 3  # num of iterations for applying scheme

    # Input for algorithm - try scheme Sn with n = 1
    s = 1
    alphas = [1]

    # prepare test data
    x = np.arange(-5, 6)
    vals = np.zeros((np.size(x), d, d))

    # Input vals - SO(d) data
    mat_fixed = np.random.rand(d, d)
    mat_fixed = mat_fixed - mat_fixed.T
    for j in range(np.size(x)):
        mu = expm((x[j] / 2) * mat_fixed)
        vals[j, :, :] = np.dot(mu, mu)

    # apply subdivision scheme and get refined values
    new_x, refined_vals = subdivision_schema(x, vals, iterations, alphas)

    # TODO perform shift in s of indices

    plot_data(new_x, refined_vals)

    return


def subdivision_schema(x, vals, iterations, alphas):
    current_vals = vals
    current_x = x

    for _ in range(iterations):
        current_x, current_vals = sn_scheme_refinement(current_x, current_vals, alphas)
    return current_x, current_vals


def sn_scheme_refinement(x, vals, alphas):
    new_len = (np.size(x) - 2) * 2 + 1
    refined_x = np.zeros(new_len)
    refined_vals = np.zeros((new_len, np.shape(vals)[1], np.shape(vals)[2]))

    # first refined point
    refined_x[0] = (x[0] + x[1])/2
    refined_vals[0, :, :] = sn_scheme_rules(vals[0:, :, :], alphas[0], is_even_indices=False)

    for j in range(2, np.size(x)):
        for alpha in alphas:
            current_index = 2 * (j - 1)
            # interpolation points
            refined_x[current_index-1] = x[j-1]
            refined_vals[current_index-1, :, :] = sn_scheme_rules(vals[j-1:, :, :], alpha, is_even_indices=True)
            # new refined points
            refined_x[current_index] = (x[j-1] + x[j]) / 2
            refined_vals[current_index, :, :] = sn_scheme_rules(vals[j-1:, :, :], alpha, is_even_indices=False)

    return refined_x, refined_vals


def sn_scheme_rules(vals, curr_alpha, is_even_indices):
    if is_even_indices:  # even indices
        new_val = vals[0, :, :]
    else:  # odd indices
        new_val = avg_func(vals[0, :, :], vals[1, :, :], curr_alpha)
    return new_val


def avg_func(a, b, alpha):
    return (1 / (1 + alpha)) * a + (alpha / (1 + alpha)) * b


def plot_data(new_x, refined_vals):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    n = np.size(new_x)

    for j in range(n):
        ax.quiver(new_x[j], 0, 0, refined_vals[j][0][0], refined_vals[j][1][0], refined_vals[j][2][0], length=0.02, color='blue', normalize=True)
        ax.quiver(new_x[j], 0, 0, refined_vals[j][0][1], refined_vals[j][1][1], refined_vals[j][2][1], length=0.02, color='red', normalize=True)
        ax.quiver(new_x[j], 0, 0, refined_vals[j][0][2], refined_vals[j][1][2], refined_vals[j][2][2], length=0.02, color='green', normalize=True)

    plt.show()

    return


if __name__ == "__main__":
    main()
