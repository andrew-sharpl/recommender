from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt

import numpy as np
import torch

#from CSC311Project.utils import *


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)
        self.encoder = nn.Sequential(nn.Linear(num_question, k), nn.Sigmoid())
        self.decoder = nn.Sequential(nn.Linear(k, num_question), nn.Sigmoid())
        # self.encoder = torch.nn.Sequential(
        #     torch.nn.Linear(num_question, k + (num_question - k)/2),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(k + (num_question - k)/2, k)
        # )
        #
        # # Building an linear decoder with Linear
        # # layer followed by Relu activation function
        # # The Sigmoid activation function
        # # outputs the value between 0 and 1
        # # 9 ==> 784
        # self.decoder = torch.nn.Sequential(
        #     torch.nn.Linear(k, k + (num_question - k)/2),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(k + (num_question - k)/2, num_question),
        #     torch.nn.Sigmoid()
        # )

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, k):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    valids = []
    trains = []
    print(valid_data.keys())
    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            if lamb is not None: loss += model.get_weight_norm()*(lamb/2)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()
        trains.append(train_loss)
        valid_acc, _ = evaluate(model, zero_train_data, valid_data)
        valids.append(valid_acc)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    plt.plot(range(num_epoch), trains, label='train loss based on epoch')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('Train Loss')
    plt.title('Train Loss v. Epoch')
    plt.savefig(f'neural_net_train_loss_{k}.png')
    plt.close()
    plt.plot(range(num_epoch), valids, label='validation accuracy based on epoch')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy v. Epoch')
    plt.savefig(f'neural_net_valid_acc_{k}.png')
    plt.close()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    predictions = []
    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        predictions.append(guess)
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total), predictions


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    valid_data_arr = []
    k_vals = [10, 50, 100, 200, 500]
    for k in k_vals:
        model = AutoEncoder(train_matrix.shape[1], k)

        # Set optimization hyperparameters.
        lr = 0.05
        num_epoch = 10
        lamb = None

        train(model, lr, lamb, train_matrix, zero_train_matrix,
              valid_data, num_epoch, k)
        valid_acc, _ = evaluate(model, zero_train_matrix, valid_data)
        valid_data_arr.append(valid_acc)
        test_acc, _ = evaluate(model, zero_train_matrix, test_data)
        print(f"test acc for k={k}: {test_acc}")
    # (c)
    plt.plot(k_vals, valid_data_arr, label='validation accuracy based on k')
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Data Accuracy v. K')
    plt.savefig('neural_net_valid_performance.png')
    best_k = k_vals[valid_data_arr.index(max(valid_data_arr))]
    print(f"k* = {best_k} with accuracy of {max(valid_data_arr)}")
    plt.close()
    valid_data_arr = []
    lambs = [0.001, 0.01, 0.1, 1]
    lr = 0.05
    num_epoch = 10
    for lamb in lambs:
        model = AutoEncoder(train_matrix.shape[1], 50)

        train(model, lr, lamb, train_matrix, zero_train_matrix,
              valid_data, num_epoch, 50)
        valid_acc, _ = evaluate(model, zero_train_matrix, valid_data)
        valid_data_arr.append(valid_acc)
    # (d)
    plt.plot(lambs, valid_data_arr, label='validation accuracy based on lambda')
    plt.legend()
    plt.xlabel('lambda')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Data Accuracy v. lambda')
    plt.savefig('neural_net_valid_performance_lambda.png')
    best_lamb = lambs[valid_data_arr.index(max(valid_data_arr))]
    model = AutoEncoder(train_matrix.shape[1], 50)
    train(model, lr, best_lamb, train_matrix, zero_train_matrix,
          valid_data, num_epoch, 50)
    test_acc, _ = evaluate(model, zero_train_matrix, test_data)
    print(f"lamb* = {best_lamb} with validation accuracy of {max(valid_data_arr)}")
    print(f"test accuracy using best lamba ({best_lamb}) = {test_acc}")
    plt.close()


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
