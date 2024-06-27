import numpy as np
from CSC311Project.part_a.item_response import *
from CSC311Project.part_b.data_cleaning import *

def update_theta_gamma(data, lr, theta, omega):
    # Piazza @824 comments mention that simultaneous updates are acceptable
    beta_partial = np.zeros(len(omega), )
    theta_partial = np.zeros(len(theta), )
    for i, q in enumerate(data["subject_id"]):
        u = data["user_id"][i]
        c_ij = data["is_proficient"][i]
        theta_i = theta[u]
        beta_j = omega[q]
        theta_partial[u] += c_ij - np.exp(theta_i) / (np.exp(theta_i) + np.exp(beta_j))
        beta_partial[q] -= c_ij - np.exp(theta_i) / (np.exp(theta_i) + np.exp(beta_j))
    # Update beta and theta:
    theta += lr * theta_partial
    omega += lr * beta_partial
    return theta, omega
#
# def update_theta_beta_lambda(data, lr, theta_matrix, omega, lamb):
#     # Piazza @824 comments mention that simultaneous updates are acceptable
#     beta_partial = np.zeros(len(omega), )
#     theta_partial = np.zeros(len(theta), )
#     for i, q in enumerate(data["question_id"]):
#         u = data["user_id"][i]
#         c_ij = data["is_proficient"][i]
#         theta_i = theta[u]
#         beta_j = omega[q]
#         theta_partial[u] += c_ij - np.exp(theta_i) / (np.exp(theta_i) + np.exp(beta_j))
#         beta_partial[q] -= c_ij - np.exp(theta_i) / (np.exp(theta_i) + np.exp(beta_j))
#     # Update beta and theta:
#     theta += lr * theta_partial
#     omega += lr * beta_partial
#     return theta, omega

def irt_1(data, val_data, lr, iterations):
    theta = np.ones(len(data["user_id"]), )
    gamma = np.ones(len(data["user_id"]), )

    val_acc_lst = []
    training_loglike = []
    val_loglike = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=gamma)
        training_loglike.append(-neg_lld)
        val_loglike.append(-neg_log_likelihood(val_data, theta=theta, beta=gamma))
        score = evaluate(data=val_data, theta=theta, beta=gamma)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, gamma = update_theta_gamma(data, lr, theta, gamma)

    return theta, gamma, val_acc_lst, training_loglike, val_loglike

# def irt_2(data, val_data, lr, iterations, theta_matrix, omega):
#     theta = np.ones(len(data["user_id"]), )
#     gamma = np.ones(len(data["user_id"]), )
#     lambd = np.ones(len(theta_matrix), len(theta_matrix[0]))
#     for j in range(len(lambd)):
#         for k in range(len(lambd[0])):
#             lambd[j][k] = omega[k]
#     beta = np.ones(len(data["user_id"]), )
#     for i in range(len(data["user_id"])):
#         beta[i] = 1/4(len(data["user_id"]))
#
#
#
#     val_acc_lst = []
#     training_loglike = []
#     val_loglike = []
#
#     for i in range(iterations):
#         neg_lld = neg_log_likelihood(data, theta=theta, beta=gamma)
#         training_loglike.append(-neg_lld)
#         val_loglike.append(-neg_log_likelihood(val_data, theta=theta, beta=gamma))
#         score = evaluate(data=val_data, theta=theta, beta=gamma)
#         val_acc_lst.append(score)
#         print("NLLK: {} \t Score: {}".format(neg_lld, score))
#         theta, gamma = update_theta_gamma(data, lr, theta, gamma)
#
#     return theta, gamma, val_acc_lst, training_loglike, val_loglike

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
    for i, q in enumerate(data["subject_id"]):
        u = data["user_id"][i]
        c_ij = data["is_proficient"][i]
        theta_i = theta[u]
        beta_j = beta[q]
        log_lklihood += c_ij * (theta_i - beta_j) - np.log(1 + np.exp((theta_i - beta_j)))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood

def main():
    train_data, test_data, valid_data = make_proficiency_data(0.5)
    iterations = 50
    theta, gamma, val_acc_lst, training_loglike, val_loglike = irt_1(train_data, valid_data, 0.01, iterations)
    acc = evaluate(test_data, theta, gamma)
    print(f"test accuracy after first training = {acc}")

if __name__=="__main__":
    main()

