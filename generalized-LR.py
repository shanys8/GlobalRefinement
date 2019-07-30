import numpy as np
from scipy.linalg import expm, logm
import matplotlib.pyplot as plt


#######################################################################################################################
# Running test data - sampled from continuous function a subdivision using Bspline with mask from lecture 6,
# 3 iterations with SO averaging
#######################################################################################################################


# 1. instead of bspline - take the mask of the scheme that I pick and extract alpha_i from it (-alpha_i ^-1  are the roots of the symbol)
# 2. each iteration j I take weighted avg func  (input: a, b, alpha_i  output: (1/(1+alpha_i))* a + (alpha_i/(1+alpha_i))* b )
# how to define s


def main():
    d = 3  # dimension of matrix input data
    iterations = 3  # num of iterations for applying scheme
    s = 1


    # prepare test data
    x = np.arange(-5, 6)
    vals = np.zeros((np.size(x), d, d))

    # check range of rand
    mat_fixed = np.random.rand(d, d)

    # test
    # mat_fixed = np.array([[0.5502,    0.2077,    0.2305], [0.6225,    0.3012,    0.8443], [0.5870,   0.4709,    0.1948]])

    mat_fixed = mat_fixed - mat_fixed.T

    for j in range(np.size(x)):
        mu = expm((x[j] / 2) * mat_fixed)
        mu = np.dot(mu, mu)
        vals[j, :, :] = mu

    # apply subdivision scheme and get refined values
    new_x, refined_vals = subdivision_schema(x, vals, iterations)

    plot_data(x, new_x, refined_vals)

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
    if is_even_indices:  # even indices
        one_avg = avg_func(vals[0, :, :], vals[2, :, :], 0.5)
        new_val = avg_func(vals[1, :, :], one_avg, 0.25)
    else:  # odd indices
        new_val = avg_func(vals[0, :, :], vals[1, :, :], 0.5)
    return new_val


def avg_func(a, b, w):
    return (1 - w) * a + w * b
    # return np.dot(a, expm(w * logm(np.dot(a.T, b))))


def plot_data(x, new_x, refined_vals):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    n = np.size(new_x)
    # x, y, z = new_x, np.zeros(n), np.zeros(n)

    # # Color by azimuthal angle
    # c = np.arctan2(v, u)
    # # Flatten and normalize
    # c = (c.ravel() - c.min()) / c.ptp()
    # # Repeat for each body line and two head lines
    # c = np.concatenate((c, np.repeat(c, 2)))
    # # Colormap
    # c = plt.cm.hsv(c)

    for j in range(n):
        ax.quiver(new_x[j], 0, 0, refined_vals[j][0][0], refined_vals[j][1][0], refined_vals[j][2][0], length=0.02, color='blue', normalize=True)
        ax.quiver(new_x[j], 0, 0, refined_vals[j][0][1], refined_vals[j][1][1], refined_vals[j][2][1], length=0.02, color='red', normalize=True)
        ax.quiver(new_x[j], 0, 0, refined_vals[j][0][2], refined_vals[j][1][2], refined_vals[j][2][2], length=0.02, color='green', normalize=True)

    plt.show()

    return


if __name__ == "__main__":
    main()
