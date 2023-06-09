from principal_RBM_alpha import *


def softmax(x):
    return (np.exp(x).T / np.sum(np.exp(x), axis=1)).T


def calcul_softmax(rbm, X):
    return softmax(np.dot(X, rbm["W"]) + rbm["b"])


def init_DBN(sizes):
    num_layers = len(sizes)
    dbn = []
    for i in range(num_layers-1):
        dbn.append(init_RBM(sizes[i], sizes[i+1]))
    return dbn


def dataset_mapper(dataset, function):
    data = dataset.data
    data = data.reshape((data.shape[0], -1))
    dataset.data = function(data)
    return dataset


def train_DBN(dbn, num_iterations, learning_rate, batch_size, ds, print_every=100):
    num_layers = len(dbn)
    for i in range(num_layers):
        print(f"Training RBM {i+1}")
        dbn[i] = train_RBM(dbn[i], epochs=num_iterations,
                           learning_rate=learning_rate,
                           batch_size=batch_size, ds=ds,
                           print_every=print_every)
        ds = dataset_mapper(ds, lambda x: entree_sortie_RBM(dbn[i], x))
    return dbn


def generer_image_DBN(dbn, num_iterations, num_images):
    for _ in range(num_images):
        x = np.random.binomial(1, 0.5, size=dbn[0]['a'].shape[0])
        for _ in range(num_iterations):
            for idx in range(len(dbn)):
                rbm = dbn[idx]
                x = entree_sortie_RBM(rbm, x)
                x = np.random.binomial(n=1, p=x)

            for idx in range(len(dbn)):
                rbm = dbn[len(dbn)-1-idx]
                x = sortie_entree_RBM(rbm, x)
                x = np.random.binomial(n=1, p=x)

        plt.imshow(x.reshape(20, 16), cmap='gray')
        plt.show()
