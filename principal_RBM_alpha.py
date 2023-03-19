from data.alphadigits import BinaryAlphaDigitsDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

def lire_alpha_digit(restricted_labels=False):
    return BinaryAlphaDigitsDataset(restrict_labels=restricted_labels).data

def init_RBM(p, q):
    return {
        'W': np.random.normal(loc=0.0, scale=0.01, size=(p, q)),
        'a' : np.zeros(shape=(1, p)),
        'b' : np.zeros(shape=(1, q))
    }

def entree_sortie_RBM(rbm, input):
    output = np.dot(input, rbm['W']) + rbm['b']
    output = 1.0 / (1.0 + np.exp(-output))
    return output

def sortie_entree_RBM(rbm, output):
    input = np.dot(output, rbm['W'].T) + rbm['a']
    input = 1.0 / (1.0 + np.exp(-input))
    return input

def train_RBM(rbm, epochs = 2001, batch_size=4, learning_rate = 1e-3, ds=BinaryAlphaDigitsDataset()):
    for epoch in range(epochs):
        errors = []
        for batch in DataLoader(ds, batch_size=batch_size, shuffle=True):
            batch = batch['data'].numpy().reshape(-1, 320)
            
            output_probs = entree_sortie_RBM(rbm, batch)
            output_hidden_states = np.random.binomial(n=1, p=output_probs)

            input_probs = sortie_entree_RBM(rbm, output_hidden_states)
            input_hidden_probs = entree_sortie_RBM(rbm, input_probs)
            input_hidden_states = np.random.binomial(n=1, p=input_hidden_probs)

            dW = np.dot(batch.T, output_probs) - np.dot(input_probs.T, input_hidden_probs)
            da = np.sum(batch - input_probs, axis=0, keepdims=True)
            db = np.sum(output_probs - input_hidden_probs, axis=0, keepdims=True)

            rbm['W'] += learning_rate * dW / batch_size
            rbm['a'] += learning_rate * da / batch_size
            rbm['b'] += learning_rate * db / batch_size

            error = np.mean((batch - input_probs) ** 2)
            errors.append(error)

        errors = np.array(errors).mean()
            
        if epoch%10==0:
            print(f"Epoch {epoch+1} - Reconstruction error: {errors}")

def generer_image_RBM(rbm, num_iterations, num_images):
    for i in range(num_images):
        v = np.random.rand(rbm['a'].shape[1])
        for j in range(num_iterations):
            h = entree_sortie_RBM(rbm, v)
            v = sortie_entree_RBM(rbm, h)
        plt.imshow(v.reshape(16, 20), cmap='gray')
        plt.show()