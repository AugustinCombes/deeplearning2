from data.alphadigits import BinaryAlphaDigitsDataset
from torch.utils.data import DataLoader
from principal_RBM_alpha import lire_alpha_digit, init_RBM, entree_sortie_RBM, sortie_entree_RBM, train_RBM, generer_image_RBM
from principal_DBN_alpha import init_DBN, train_DBN, generer_image_DBN
import numpy as np


def init_DNN(sizes):
    dnn = init_DBN(sizes)
    return dnn


def calcul_softmax(rbm, input):
    output = entree_sortie_RBM(rbm, input)
    probas = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
    return probas


def entree_sortie_reseau(dnn, input):
    outputs = []
    for rbm in dnn[:-1]:
        output = entree_sortie_RBM(rbm, input)
        outputs.append(output)
        input = output

    probas = calcul_softmax(dnn[-1], input)
    outputs.append(probas)

    return outputs


def pretrain_DNN(dnn, num_iterations, learning_rate, batch_size, ds):
    sub_dbn = dnn[:-1]
    sub_dbn = train_DBN(sub_dbn, num_iterations,
                        learning_rate, batch_size, ds, print_every=1)
    dnn = sub_dbn + dnn[-1:]
    return dnn


def retropropagation(dnn, num_iterations, learning_rate, batch_size, ds, print_every=1):
    for epoch in range(num_iterations):
        epoch_loss = []
        for batch in DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True):
            batch_data = batch['data'].numpy().reshape(batch_size, -1)
            batch_labels = batch['labels']

            # Propagation avant
            hidden_outputs = entree_sortie_reseau(dnn, batch_data)
            predicted_labels = np.argmax(hidden_outputs[-1], axis=1)

            # Calcul de l'entropie croisée et de la perte
            labels_one_hot = np.eye(dnn[-1]['W'].shape[1])[batch_labels]
            output = hidden_outputs[-1]
            cross_entropy = - \
                np.mean(np.sum(labels_one_hot * np.log(output), axis=1))
            epoch_loss.append(cross_entropy)

            # Propagation arrière
            delta = (output - labels_one_hot) / batch_size

            # Mise à jour du dernier RBM (couche de classification)
            rbm = dnn[-1]
            dW = np.dot(hidden_outputs[-2].T, delta)
            db = np.mean(delta, axis=0, keepdims=True)
            rbm['W'] -= learning_rate * dW
            rbm['b'] -= learning_rate * db

            # Mise à jour des couches cachées
            for i in range(len(dnn)-2, -1, -1):
                rbm = dnn[i]
                if i == 0:
                    prev_hidden_output = batch_data
                else:
                    prev_hidden_output = hidden_outputs[i - 1]

                prev_delta = np.dot(delta, dnn[i + 1]['W'].T)
                delta_input = prev_delta * \
                    hidden_outputs[i] * (1 - hidden_outputs[i])
                dW = np.dot(prev_hidden_output.T, delta_input)
                db = np.mean(delta_input, axis=0, keepdims=True)
                rbm['W'] -= learning_rate * dW
                rbm['b'] -= learning_rate * db
                delta = delta_input

        epoch_loss = np.mean(epoch_loss)

        if epoch % print_every == 0:
            print(f"Epoch {epoch+1} - Loss: {epoch_loss}")

    return dnn
