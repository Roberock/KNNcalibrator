import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def scatter_post(ax, theta, truth=None, title="", alpha=0.30, s=6, label="Posterior"):
    ax.scatter(theta[:,0], theta[:,1], s=s, alpha=alpha, label=label)
    if truth is not None:
        ax.scatter(truth[:,0], truth[:,1], c="r", marker="x", s=60, label="θ true cloud")
    ax.set_title(title); ax.set_xlabel("θ1"); ax.set_ylabel("θ2"); ax.grid(True); ax.legend()


def plot_kde_2d(samples, true_theta=None, gridsize=100, ax=None):
    kde = gaussian_kde(samples.T)

    # grid
    x = np.linspace(samples[:,0].min(), samples[:,0].max(), gridsize)
    y = np.linspace(samples[:,1].min(), samples[:,1].max(), gridsize)
    X, Y = np.meshgrid(x, y)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    if ax is None:
        plt.contourf(X, Y, Z, colors='b', levels=30, alpha=0.7)
        plt.contour(X, Y, Z, colors='k', linewidths=0.5, label = 'kde')
        if true_theta is not None:
            plt.scatter(true_theta[:,0], true_theta[:,1], c='r', s=40, label = 'target')

        plt.xlabel(r'$\theta_1$')
        plt.ylabel(r'$\theta_2$')
        plt.legend()
        plt.tight_layout()
    else:
        ax.contour(X, Y, Z, colors='k', linewidths=0.5)



def weighted_mixture_kde(posterior_list, weights, gridsize=100,
                         true_theta=None):

    kdes = [gaussian_kde(D.T) for D in posterior_list]

    # construct global grid
    all_samples = np.vstack(posterior_list)
    x = np.linspace(all_samples[:,0].min(), all_samples[:,0].max(), gridsize)
    y = np.linspace(all_samples[:,1].min(), all_samples[:,1].max(), gridsize)
    X, Y = np.meshgrid(x, y)
    pts = np.vstack([X.ravel(), Y.ravel()])

    # mixture evaluation
    Z = np.zeros(pts.shape[1])
    for w, kde in zip(weights, kdes):
        Z += w * kde(pts)

    Z = Z.reshape(X.shape)

    plt.contourf(X, Y, Z, colors='b', levels=30, alpha=0.7)
    plt.contour(X, Y, Z, colors='k', linewidths=0.5)

    if true_theta is not None:
        plt.scatter(true_theta[:,0], true_theta[:,1], c='r', s=40)

    plt.tight_layout()
    plt.show()
