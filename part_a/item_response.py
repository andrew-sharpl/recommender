from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        c_ij = data["is_correct"][i]
        theta_i = theta[u]
        beta_j = beta[q]
        log_lklihood += c_ij * (theta_i - beta_j) - np.log(1 + np.exp((theta_i - beta_j)))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    # Piazza @824 comments mention that simultaneous updates are acceptable
    beta_partial = np.zeros(len(beta), )
    theta_partial = np.zeros(len(theta), )
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        c_ij = data["is_correct"][i]
        theta_i = theta[u]
        beta_j = beta[q]
        theta_partial[u] += c_ij - np.exp(theta_i) / (np.exp(theta_i) + np.exp(beta_j))
        beta_partial[q] -= c_ij - np.exp(theta_i) / (np.exp(theta_i) + np.exp(beta_j))
    # Update beta and theta:
    theta += lr * theta_partial
    beta += lr * beta_partial

    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    theta = np.ones(len(set(data["user_id"])), )
    beta = np.ones(len(set(data["question_id"])), )

    val_acc_lst = []
    training_loglike = []
    val_loglike = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        training_loglike.append(-neg_lld)
        val_loglike.append(-neg_log_likelihood(val_data, theta=theta, beta=beta))
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta, val_acc_lst, training_loglike, val_loglike


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    iterations = 50
    theta, beta, val_acc_lst, training_loglike, val_loglike = irt(train_data, val_data, 0.01, iterations)
    acc = evaluate(test_data, theta, beta)
    print(acc)

    # Plotting:
    fig, axs = plt.subplots(2)
    fig.suptitle('Log-Likelihood vs # Iterations')
    axs[0].set_title("Validation Set")
    axs[0].set_xlabel("Iterations")
    axs[0].set_ylabel("Log-Likelihood")

    axs[1].set_title("Training Set")
    axs[1].set_xlabel("Iterations")
    axs[1].set_ylabel("Log-Likelihood")

    axs[0].plot(val_loglike, color='red')
    axs[1].plot(training_loglike, color='blue')
    plt.subplots_adjust(bottom=0.1,
                        top=0.840,
                        hspace=0.5)
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # Implement part (d)                                                #
    theta_scale = [i/2 for i in range(10)]
    prob_curve1 = [sigmoid(theta_scale[i] - beta[0]) for i in range(10)]
    prob_curve2 = [sigmoid(theta_scale[i] - beta[2]) for i in range(10)]
    prob_curve3 = [sigmoid(theta_scale[i] - beta[4]) for i in range(10)]
    plt.title("Probability of Correctness vs Student Ability (Theta)")
    plt.xlabel("Theta")
    plt.ylabel("Probability of Correctness")
    plt.plot(theta_scale, prob_curve1, color='red', label='Question 1525')
    plt.plot(theta_scale, prob_curve2, color='blue', label='Question 1030')
    plt.plot(theta_scale, prob_curve3, color='green', label='Question 773')
    plt.legend(loc='upper center', bbox_to_anchor=(0.8, 0.3), shadow=True, ncol=1)
    plt.show()
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
