import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from skimage import measure
from sklearn.datasets import make_classification
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def plot_hyperplane(clf, ax: Axes = None, interval: float = .05, alpha=.3,
                    colors: list = ('r', 'b')) -> Line3DCollection:
    """
Plots the hyperplane of the model in an axes
    :param clf: the classifier to use to find the hyperplane
    :param ax: the axes to plot the hyperplane into
    :param interval: the precision of the the hyperplane rendering.
    :return: the mesh of the created hyperplane that was added to the axes
    """

    is_3d = False

    if ax is None:
        try:
            clf.predict([[0, 0, 0]])
            is_3d = True
            ax = plt.gca(projection="3d")
        except ValueError:
            is_3d = False
            ax = plt.gca()

    elif isinstance(ax, Axes3D):
        is_3d = True

    interval = int(1 / interval)

    # get the separating hyperplane
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(x_min, x_max, interval)
    yy = np.linspace(y_min, y_max, interval)

    if is_3d:
        z_min, z_max = ax.get_zlim()

        zz = np.linspace(z_min, z_max, interval)

        yy, xx, zz = np.meshgrid(yy, xx, zz)

        if hasattr(clf, "decision_function"):
            z = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
        elif hasattr(clf, "predict_proba"):
            z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])[:, 1]
        z = z.reshape(xx.shape)

        vertices, faces, _, _ = measure.marching_cubes(z, 0)
        # Scale and transform to actual size of the interesting volume
        vertices = vertices * [x_max - x_min, y_max - y_min, z_max - z_min] / interval
        vertices += [x_min, y_min, z_min]
        # and create a mesh to display
        mesh = Line3DCollection(vertices[faces],
                                facecolor=colors, alpha=alpha)

        ax.add_collection3d(mesh)

        return mesh
    else:
        xx, yy = np.meshgrid(xx,
                             yy)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        return ax.contourf(xx, yy, Z, 10, colors=colors, alpha=alpha)


def plot_fitting_plane(clf, ax, number: int = 50, color=None):
    x_min, x_max = ax.get_xlim()

    if isinstance(ax, Axes3D):
        y_min, y_max = ax.get_ylim()
        yy, xx = np.meshgrid(np.linspace(y_min, y_max, number), np.linspace(x_min, x_max, number))
        z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)

        return ax.plot_wireframe(xx, yy, z, facecolors=color)
    else:
        x = np.linspace(x_min, x_max, number)
        y = clf.predict(x)

        return ax.plot(x, y, c=color)


def test_classification_data(n_samples=100, n_features=3, random_state=None) -> (np.ndarray, np.ndarray):
    return make_classification(n_samples, n_features=n_features, n_informative=n_features, n_redundant=0,
                               random_state=random_state)


def create_color_array(labels, colors_array):
    return np.array(list(map(lambda x: colors_array[x], labels)))


def retrieve_svm_features_manipulated_by_kernel(svm, features):
    return features * svm._compute_kernel(features)


def deep_grid_search_cv_svm(clf, features, labels, c_min=1, c_max=1e3, c_precision=1000, gamma_min=1e-2, gamma_max=1e2,
                            gamma_precision=1000, kernels=("linear", "rbf", "sigmoid"),
                            class_weights=("balanced", None)):
    gamma_test = ["auto"]
    gamma_test.extend(list(np.linspace(gamma_min, gamma_max, gamma_precision)))

    param_grid = {"C": np.linspace(c_min, c_max, c_precision),
                  "gamma": gamma_test,
                  "kernel": kernels,
                  "class_weight": class_weights}

    grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, verbose=10)
    grid_search.fit(features, labels)

    return grid_search


def save_sklearn_model(clf, filename):
    return joblib.dump(clf, filename)


def load_sklean_model(filename):
    return joblib.load(filename)


def example():
    features, labels = test_classification_data(n_features=2)

    clf = SVC()
    clf.fit(features, labels)

    color = ['r', 'b']
    plt.scatter(features[:, 0], features[:, 1], c=create_color_array(labels, color))
    plot_hyperplane(clf, interval=.001, colors=color)

    plt.show()


if __name__ == '__main__':
    example()
