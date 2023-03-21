from data.alphadigits import BinaryAlphaDigitsDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


def lire_alpha_digit(restricted_labels=False):
    return BinaryAlphaDigitsDataset(restrict_labels=restricted_labels).data


def init_RBM(p, q):
    return {
        'W': np.random.normal(loc=0.0, scale=0.01, size=(p, q)),
        'a': np.zeros(shape=(1, p)),
        'b': np.zeros(shape=(1, q))
    }


def entree_sortie_RBM(rbm, input):
    output = np.dot(input, rbm['W']) + rbm['b']
    output = 1.0 / (1.0 + np.exp(-output))
    return output


def sortie_entree_RBM(rbm, output):
    input = np.dot(output, rbm['W'].T) + rbm['a']
    input = 1.0 / (1.0 + np.exp(-input))
    return input


def train_RBM(rbm, epochs=2001, batch_size=4, learning_rate=1e-1, ds=BinaryAlphaDigitsDataset()):
    for epoch in range(epochs):
        errors = []
        for batch in DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True):
            batch = batch['data'].numpy()
            batch = batch.reshape(batch.shape[0], -1)

            output_probs = entree_sortie_RBM(rbm, batch)
            output_hidden_states = np.random.binomial(n=1, p=output_probs)

            input_probs = sortie_entree_RBM(rbm, output_hidden_states)
            input_hidden_probs = entree_sortie_RBM(rbm, input_probs)
            input_hidden_states = np.random.binomial(n=1, p=input_hidden_probs)

            dW = np.dot(batch.T, output_probs) - \
                np.dot(input_probs.T, input_hidden_probs)
            da = np.sum(batch - input_probs, axis=0, keepdims=True)
            db = np.sum(output_probs - input_hidden_probs,
                        axis=0, keepdims=True)

            rbm['W'] += learning_rate * dW / batch_size
            rbm['a'] += learning_rate * da / batch_size
            rbm['b'] += learning_rate * db / batch_size

            error = np.mean((batch - input_probs) ** 2)
            errors.append(error)

        errors = np.array(errors).mean()

        if epoch % 100 == 0:
            print(f"Epoch {epoch+1} - Reconstruction error: {errors}")

    return rbm


def generer_image_RBM(rbm, num_iterations, num_images):
    for _ in range(num_images):
        v = np.random.binomial(1, 0.5, size=rbm['a'].shape[1])
        for _ in range(num_iterations):
            h = entree_sortie_RBM(rbm, v)
            h = np.random.binomial(n=1, p=h)
            v = sortie_entree_RBM(rbm, h)
            v = np.random.binomial(n=1, p=v)
        plt.imshow(v.reshape(20, 16), cmap='gray')
        plt.show()
