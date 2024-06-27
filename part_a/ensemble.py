from utils import _load_csv
from utils import *
from utils import evaluate as pred_eval
from numpy import random
from scipy.sparse import dok_matrix
import numpy as np
from part_a.knn import knn_impute_by_item, knn_impute_by_user
from part_a.item_response import *
from part_a.neural_network import *


# TODO: complete this file.
def csv_bootstrap(file, n_samples):
    data = _load_csv(file)
    return bootstrap(data, n_samples)


def bootstrap(data, n_samples):
    num_entries = len(data["user_id"])

    bootstrap_sample = {
        "question_id": [],
        "user_id": [],
        "is_correct": []
    }

    # random indices with replacement
    indices = random.randint(num_entries, size=n_samples)

    for i in indices:
        question_id = data["question_id"][i]
        user_id = data["user_id"][i]
        is_correct = data["is_correct"][i]

        bootstrap_sample["question_id"].append(question_id)
        bootstrap_sample["user_id"].append(user_id)
        bootstrap_sample["is_correct"].append(is_correct)

    return bootstrap_sample
def knn_predictions(data, val_data, k, type):
    unique_users = len(set(data["user_id"]))
    unique_questions = len(set(data["question_id"]))
    sparse_matrix = dok_matrix((unique_users, unique_questions), dtype=np.float32)

    for i in range(unique_users):
        for j in range(unique_questions):
            sparse_matrix[i, j] = np.NAN

    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        cur_is_correct = data["is_correct"][i]
        sparse_matrix[cur_user_id, cur_question_id] = cur_is_correct

    sparse_matrix = sparse_matrix.toarray()
    acc, pred_mat = None, None
    if type == "item":
        acc, pred_mat = knn_impute_by_item(sparse_matrix, val_data, k)
    elif type == "user":
        acc, pred_mat = knn_impute_by_user(sparse_matrix, val_data, k)

    return sparse_matrix_predictions(val_data, pred_mat)


def nn_predictions(lr, k, epochs, lamb, data, val_data):
    unique_users = len(set(data["user_id"]))
    unique_questions = len(set(data["question_id"]))
    train_matrix = dok_matrix((unique_users, unique_questions), dtype=np.float32)

    # filling matrix with NAN
    for i in range(unique_users):
        for j in range(unique_questions):
            train_matrix[i, j] = np.NAN

    # filling matrix with data
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        cur_is_correct = data["is_correct"][i]
        train_matrix[cur_user_id, cur_question_id] = cur_is_correct

    train_matrix = train_matrix.toarray()
    zero_train_matrix = train_matrix.copy()
    zero_train_matrix[np.isnan(train_matrix)] = 0
    train_matrix = torch.FloatTensor(train_matrix)
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)

    model = AutoEncoder(train_matrix.shape[1], k)

    train(model, lr, lamb, train_matrix, zero_train_matrix, val_data, epochs, k)
    acc, predicitons = evaluate(model, zero_train_matrix, val_data)
    return predicitons


def prob_predictions(iterations, learning_rate, data, val_data):
    theta, beta, val_acc_lst, training_loglike, val_loglike = irt(data, val_data, learning_rate, iterations)
    predictions = []

    for i, q in enumerate(val_data["question_id"]):
        u = val_data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        predictions.append(p_a >= 0.5)

    return predictions


def ensemble_predictions(knn, nn, prob):
    assert (len(knn) == len(nn) == len(prob))
    predictions = []
    for i in range(len(knn)):
        n = knn[i] + nn[i] + prob[i]
        if n >= 2:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions


if __name__ == '__main__':
    # loading data
    train_data = load_train_csv('../data')
    val_data = load_valid_csv("../data")
    training_size = len(train_data["user_id"])
    test_data = load_public_test_csv('../data')

    # initializing bootstrap samples
    knn_bootstrap = bootstrap(train_data, training_size)
    nn_bootstrap = bootstrap(train_data, training_size)
    prob_bootstrap = bootstrap(train_data, training_size)

    # knn hyperparameters
    knn_k = 21
    knn_type = "item"

    # neural net hyperparameters
    nn_lr = 0.05
    nn_k = 100
    nn_epoch = 10
    nn_lamb = 0.01

    # probabilistic model hyperparameters
    prob_iter = 50
    prob_lr = 0.01


    # validation data accuracy
    knn_val_predictions = knn_predictions(knn_bootstrap, val_data, knn_k, knn_type)
    nn_val_predictions = nn_predictions(nn_lr, nn_k, nn_epoch, nn_lamb, nn_bootstrap, val_data)
    prob_val_predictions = prob_predictions(prob_iter, prob_lr, prob_bootstrap, val_data)
    ensemble_val_predictions = ensemble_predictions(knn_val_predictions, nn_val_predictions, prob_val_predictions)

    print("knn (validation):", pred_eval(val_data, knn_val_predictions))
    print("nn (validation):", pred_eval(val_data, nn_val_predictions))
    print("prob (validation)", pred_eval(val_data, prob_val_predictions))
    print("ensemble (validation)", pred_eval(val_data, ensemble_val_predictions))

    # test data accuracy
    knn_test_predictions = knn_predictions(knn_bootstrap, test_data, knn_k, knn_type)
    nn_test_predictions = nn_predictions(nn_lr, nn_k, nn_epoch, nn_lamb, nn_bootstrap, test_data)
    prob_test_predictions = prob_predictions(prob_iter, prob_lr, prob_bootstrap, test_data)
    ensemble_test_predictions = ensemble_predictions(knn_test_predictions, nn_test_predictions, prob_test_predictions)

    print("knn (test):", pred_eval(test_data, knn_test_predictions))
    print("nn (test):", pred_eval(test_data, nn_test_predictions))
    print("prob (test)", pred_eval(test_data, prob_test_predictions))
    print("ensemble (test)", pred_eval(test_data, ensemble_test_predictions))
