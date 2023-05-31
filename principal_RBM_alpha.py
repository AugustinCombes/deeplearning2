from data.alphadigits import BinaryAlphaDigitsDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def lire_alpha_digit(restricted_labels=False):
    return BinaryAlphaDigitsDataset(restrict_labels=restricted_labels).data


def init_RBM(p, q):
    return {
        'W': np.random.normal(loc=0.0, scale=0.01, size=(p, q)),
        'a': np.zeros(p),
        'b': np.zeros(q)
    }


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def entree_sortie_RBM(rbm, input):
    output = np.dot(input, rbm['W']) + rbm['b']
    return sigmoid(output)


def sortie_entree_RBM(rbm, output):
    input = np.dot(output, rbm['W'].T) + rbm['a']
    return sigmoid(input)


def train_RBM(rbm, epochs=2001, batch_size=4, learning_rate=1e-1, ds=BinaryAlphaDigitsDataset(),
              print_every=100):
    p,q = rbm['W'].shape

    for epoch in tqdm(range(epochs)):
        for batch in DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True):
            batch = batch['data'].numpy()
            batch = batch.reshape(batch.shape[0], -1)

            length = len(batch)
            output_probs = entree_sortie_RBM(rbm, batch)
            output_hidden_states = (np.random.uniform(size=(length, q)) < output_probs).astype("float")
            input_hidden_states = (
                np.random.uniform(size=(length, p)) < sortie_entree_RBM(rbm, output_hidden_states)
            ).astype("float")
            input_hidden_probs = entree_sortie_RBM(rbm, input_hidden_states)

            dW = np.dot(batch.T, output_probs) - np.dot(input_hidden_states.T, input_hidden_probs)
            da = np.sum(batch - input_hidden_states, axis=0)
            db = np.sum(output_probs - input_hidden_probs, axis=0)

            rbm["W"] += learning_rate*dW/length 
            rbm["a"] += learning_rate*da/length 
            rbm["b"] += learning_rate*db/length 
    
    return rbm


def generer_image_RBM(rbm, num_iterations, num_images):
    for _ in range(num_images):
        v = np.random.binomial(1, 0.5, size=rbm['a'].shape[0])
        for _ in range(num_iterations):
            h = entree_sortie_RBM(rbm, v)
            h = np.random.binomial(n=1, p=h)
            v = sortie_entree_RBM(rbm, h)
            v = np.random.binomial(n=1, p=v)
        plt.imshow(v.reshape(20, 16), cmap='gray')
        plt.show()
