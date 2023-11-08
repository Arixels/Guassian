# %%
import numpy as np
import matplotlib.pyplot as plt
def gaussian_2d(kernel_size: int, sigma_x: float = 1, sigma_y: float = 1, mu_x: float = 0, mu_y: float = 0):
    """
    Generate a 2D Gaussian kernel.

    Args:
        kernel_size (int): The size of the kernel.
        sigma_x (float): Standard deviation in the x-direction.
        sigma_y (float): Standard deviation in the y-direction.
        mu_x (float): Mean in the x-direction.
        mu_y (float): Mean in the y-direction.

    Returns:
        np.ndarray: The 2D Gaussian kernel.
    """
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    
    normal = 1 / (2 * np.pi * sigma_x * sigma_y)
    gauss = normal * np.exp(-((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2)))
    
    return gauss

def multivariate_gaussian(pos:np.ndarray, mu:np.ndarray, Sigma:np.ndarray):
    """
    Calculate the multivariate Gaussian distribution.

    Args:
        pos (np.ndarray): The position at which to evaluate the distribution.
        mu (np.ndarray): The mean vector.
        Sigma (np.ndarray): The covariance matrix.

    Returns:
        np.ndarray: The probability density values at the given position.
    """
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi)**n * Sigma_det)
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

    return np.exp(-fac / 2) / N



# %%

if __name__ == '__main__':
    kernel_size = 3
    gaussian = gaussian_2d(kernel_size)
    # gaussian = gaussian_filter_unique_means_2d(kernel_size)


    print("Gaussian filter of {} X {}:".format(kernel_size, kernel_size))
    print(gaussian)


    plt.imshow(gaussian, cmap='viridis', extent=(-1, 1, -1, 1))
    plt.title("Gaussian Filter")
    plt.colorbar()
    plt.show()

    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))

    # Create a 3D plot
    # plot using subplots
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1,projection='3d')

    ax1.plot_surface(x, y, gaussian, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap='viridis')
    ax1.view_init(55,-70)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$y$')

    ax2 = fig.add_subplot(2,1,2,projection='3d')
    ax2.contourf(x, y, gaussian, zdir='z', offset=0, cmap='viridis')
    ax2.view_init(90, 270)

    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$y$')

    plt.show()


# %%
from matplotlib import cm

if __name__ == '__main__':
    N = 9
    X = np.linspace(-1, 1, N)
    Y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(X, Y)

    # Mean vector and covariance matrix
    mu = np.array([0., 0.])
    Sigma = np.array([[ 1. , 0.5], [0.5,  1.]])

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    print(pos.shape)

    # The distribution on the variables X, Y packed into pos.
    Z = multivariate_gaussian(pos, mu, Sigma)

    # plot using subplots
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1,projection='3d')

    ax1.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap=cm.viridis)
    ax1.view_init(55,-70)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.set_xlabel(r'$x_1$')
    ax1.set_ylabel(r'$x_2$')

    ax2 = fig.add_subplot(2,1,2,projection='3d')
    ax2.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.viridis)
    ax2.view_init(90, 270)

    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax2.set_xlabel(r'$x_1$')
    ax2.set_ylabel(r'$x_2$')

    plt.show()
    print(Z)
# %%
