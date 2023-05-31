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


def softmax(x):
    return (np.exp(x).T / np.sum(np.exp(x), axis=1)).T


def calcul_softmax(layer, X):
    return softmax(np.dot(X, layer["W"]) + layer["b"])


def entree_sortie_reseau(dnn, input):
    outputs = [input]
    for rbm in dnn[:-1]:
        output = entree_sortie_RBM(rbm, outputs[-1])
        outputs.append(output)

    probas = calcul_softmax(dnn[-1], outputs[-1])
    outputs.append(probas)

    return outputs


def pretrain_DNN(dnn, num_iterations, learning_rate, batch_size, ds):
    sub_dbn = dnn[:-1]
    sub_dbn = train_DBN(sub_dbn, num_iterations,
                        learning_rate, batch_size, ds, print_every=1)
    dnn = sub_dbn + dnn[-1:]
    return dnn


def retropropagation(dnn, num_iterations, learning_rate, batch_size, ds, print_result=False):
    for epoch in range(num_iterations):
        epoch_loss = list()
        
        for batch in DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True):
            X_batch = batch['data'].numpy().reshape(batch_size, -1)
            batch_labels = batch['labels']
            y_batch = np.eye(dnn[-1]['W'].shape[1])[batch_labels]
            
            length = len(X_batch)
            sortie = entree_sortie_reseau(dnn, X_batch)

            d_Z = sortie[-1] - y_batch

            for j in range(len(dnn) - 1, -1, -1):
                dW = np.dot(sortie[j].T, d_Z)
                db = np.sum(d_Z, axis=0)

                # update W and b
                dnn[j]["W"] -= learning_rate*dW/length
                dnn[j]["b"] -= learning_rate*db/length

                if j == 0:
                    break

                d_A = np.dot(d_Z, dnn[j]["W"].T)
                d_Z = d_A * sortie[j] * (1 - sortie[j])

            cross_entropy = -np.mean(np.sum(y_batch * np.log(sortie[-1]), axis=1))
            epoch_loss.append(cross_entropy)
        
        loss = np.array(epoch_loss).mean()
        if print_result:
            print(f'Retropropagation @ epoch {epoch}: loss', "%.2f" % loss)
        
    return dnn