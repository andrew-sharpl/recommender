import numpy as np
import csv
import os
import pandas as pd
import ast
import numpy as np
from CSC311Project.part_a.item_response import *
from CSC311Project.part_b.data_cleaning import *
from CSC311Project.part_b.matrix_maker import *

def load_data(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["question_id"].append(int(row[0]))
                data["user_id"].append(int(row[1]))
                data["is_correct"].append(int(row[2]))
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def load_question_meta(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "question_id": [],
        "subject_id": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["question_id"].append(int(row[0]))
                list_str = row[1]
                subject_list = ast.literal_eval(list_str)
                data["subject_id"].append(subject_list)
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def merge_data(data, meta):
    data_df = pd.DataFrame.from_dict(data)
    meta_df = pd.DataFrame.from_dict(meta)
    joined_df = data_df.merge(meta_df, on="question_id")

    return joined_df


def create_data(data_path, out_path, meta_path='../data/question_meta.csv'):
    data = load_data(data_path)
    meta = load_question_meta(meta_path)
    joined = merge_data(data, meta)
    user_subject_dict = {
        "user_id": [],
        "subject_id": [],
        "subject_average": []
    }
    unique_user_ids = list(set(data["user_id"]))
    for user in unique_user_ids:
        user_subjects = []
        seen = set()
        user_frame = joined.loc[joined['user_id'] == user]
        for i, row in user_frame.iterrows():
            subjects = row['subject_id']
            for subject in subjects:
                if subject not in seen:
                    user_subjects.append(subject)
                    seen.add(subject)

        for i in range(len(user_subjects)):
            user_subject_dict['user_id'].append(user)
            user_subject_dict['subject_id'].append(user_subjects[i])

    for i in range(len(user_subject_dict['user_id'])):
        cur_user_id = user_subject_dict['user_id'][i]
        cur_subject_id = user_subject_dict['subject_id'][i]
        user_frame = joined.loc[joined['user_id'] == cur_user_id]
        subject_correctness = []
        for i, row in user_frame.iterrows():
            subject_lst = row['subject_id']
            if cur_subject_id in subject_lst:
                subject_correctness.append(row['is_correct'])
        subject_avg = sum(subject_correctness) / len(subject_correctness)
        user_subject_dict['subject_average'].append(subject_avg)

    usd = pd.DataFrame(user_subject_dict)
    usd.to_csv(out_path, index=False)


def avg_to_proficiency(avg_path, out_path, threshold=0.5):
    # A helper function to load the csv file.
    if not os.path.exists(avg_path):
        raise Exception("The specified path {} does not exist.".format(avg_path))
    # Initialize the data.
    data = {
        "user_id": [],
        "subject_id": [],
        "is_proficient": []
    }
    # Iterate over the row to fill in the data.
    with open(avg_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["user_id"].append(int(row[0]))
                data["subject_id"].append(int(row[1]))
                avg = float(row[2])
                if avg >= threshold:
                    data["is_proficient"].append(1)
                else:
                    data["is_proficient"].append(0)
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    df = pd.DataFrame.from_dict(data)
    df.to_csv(out_path, index=False)
    return data



def make_proficiency_data(threshold):
    train_dict = avg_to_proficiency('../data/train_sub_avg.csv', '../data/train_proficiency.csv', threshold)
    test_dict = avg_to_proficiency('../data/test_sub_avg.csv', '../data/test_proficiency.csv', threshold)
    valid_dict = avg_to_proficiency('../data/valid_sub_avg.csv', '../data/valid_proficiency.csv', threshold)
    return train_dict, test_dict, valid_dict

def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["subject_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_proficient"] == np.array(pred))) \
           / len(data["is_proficient"])

def evaluate2(data, subject_df, theta_matrix, lambda_matrix, beta):
    pred = []
    for i, u in enumerate(data["user_id"]):
        j = data["question_id"][i]
        avg_theta_lambda = 0
        subjects = list(subject_df.loc[subject_df["question_id"] == j]["subject_id"])[0]
        for k, s in enumerate(subjects):
            avg_theta_lambda += theta_matrix[u][s] * lambda_matrix[j][s]
        avg_theta_lambda /= len(subjects)
        x = avg_theta_lambda - beta[j]
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])

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

def update_theta_beta_lambda(data_original, lr, theta_matrix, lamb_matrix, beta, df):
    beta_partial = np.zeros(len(beta), )
    theta_matrix_partial = np.zeros((len(theta_matrix), len(theta_matrix[0])))
    lambda_matrix_partial = np.zeros((len(lamb_matrix), len(lamb_matrix[0])))
    for i, u in enumerate(data_original["user_id"]):
        j = data_original["question_id"][i]
        c_ij = data_original["is_correct"][i]
        avg_theta_lambda = 0
        subjects = list(df.loc[df["question_id"] == j]["subject_id"])[0]
        for k, s in enumerate(subjects):
            avg_theta_lambda += theta_matrix[u][s] * lamb_matrix[j][s]
        avg_theta_lambda /= len(subjects)
        beta_partial[j] += -c_ij + np.exp(avg_theta_lambda) / (np.exp(avg_theta_lambda) + np.exp(beta[j]))
        for k, s in enumerate(subjects):
            theta_matrix_partial[u][s] = \
                c_ij * lamb_matrix[j][s] / len(subjects) \
                + (np.exp(avg_theta_lambda) * lamb_matrix[j][s]/len(subjects))\
                / (np.exp(avg_theta_lambda) + np.exp(beta[j]))
            lambda_matrix_partial[j][s] = c_ij * theta_matrix[u][s] / len(subjects) \
                                          + (np.exp(avg_theta_lambda) * theta_matrix[u][s]/len(subjects)) \
                                          / (np.exp(avg_theta_lambda) + np.exp(beta[j]))
    theta_matrix += lr * theta_matrix_partial
    beta += lr * beta_partial
    lamb_matrix += lr * lambda_matrix_partial
    return theta_matrix, lamb_matrix, beta


def irt_1(data, val_data, lr, iterations):
    theta = np.ones(max(data["user_id"]) + 1, )
    gamma = np.ones(max(data["subject_id"]) + 1, )
    val_acc_lst = []
    training_loglike = []
    val_loglike = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta, gamma)
        training_loglike.append(-neg_lld)
        val_loglike.append(-neg_log_likelihood(val_data, theta, gamma))
        score = evaluate(val_data, theta, gamma)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, gamma = update_theta_gamma(data, lr, theta, gamma)

    return theta, gamma, val_acc_lst, training_loglike, val_loglike

def irt_2(train_data, subject_df,  val_data, lr, iterations, theta, omega):
    print("creating matrices")
    theta_matrix, lambda_matrix, beta = create_matrices(theta, omega, train_data, subject_df)
    val_acc_lst = []
    training_loglike = []
    val_loglike = []
    for i in range(iterations):
        print(f"iteration {i}")
        neg_lld = neg_log_liklihood_2(train_data, subject_df, theta_matrix, lambda_matrix, beta)
        training_loglike.append(-neg_lld)
        val_loglike.append(neg_log_liklihood_2(val_data, subject_df, theta_matrix, lambda_matrix, beta))
        score = evaluate2(val_data, subject_df, theta_matrix, lambda_matrix, beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta_matrix, lambda_matrix, beta = update_theta_beta_lambda(train_data, lr, theta_matrix, lambda_matrix, beta, subject_df)

    return theta_matrix, lambda_matrix, beta, val_acc_lst, training_loglike, val_loglike

def main():
    train_data, test_data, valid_data = make_proficiency_data(0.5)
    iterations = 50
    lr = 0.01
    theta, omega, val_acc_lst, training_loglike, val_loglike = irt_1(train_data, valid_data, lr, iterations)
    acc = evaluate(test_data, theta, omega)
    print(f"test accuracy after first training = {acc}")
    theta = sigmoid(theta)
    omega = sigmoid(omega)
    train_data = load_train_csv("../data")
    valid_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    meta = load_question_meta("../data/question_meta.csv")
    df = pd.DataFrame.from_dict(meta)
    theta_matrix, lambda_matrix, beta, val_acc_lst, training_loglike, val_loglike = irt_2(train_data, df, valid_data, lr, iterations, theta, omega)
    acc_final = evaluate2(test_data, df, theta_matrix, lambda_matrix, beta)
    print(f"final test accuracy after second training = {acc_final}")

def neg_log_likelihood(data, theta, omega):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    log_lklihood = 0.
    for i, q in enumerate(data["subject_id"]):
        u = data["user_id"][i]
        c_ij = data["is_proficient"][i]
        theta_i = theta[u]
        omega_j = omega[q]
        log_lklihood += c_ij * (theta_i - omega_j) - np.log(1 + np.exp((theta_i - omega_j)))
    return -log_lklihood

def neg_log_liklihood_2(data_original, df, theta_matrix, lamb_matrix, beta):
    log_liklihood = 0.
    for i, u in enumerate(data_original["user_id"]):
        j = data_original["question_id"][i]
        c_ij = data_original["is_correct"][i]
        avg_theta_lambda = 0
        subjects = list(df.loc[df["question_id"] == j]["subject_id"])[0]
        for k, s in enumerate(subjects):
            avg_theta_lambda += theta_matrix[u][s] * lamb_matrix[j][s]
        avg_theta_lambda /= len(subjects)
        log_liklihood += c_ij * (avg_theta_lambda - beta[j]) - np.log(1 + np.exp(avg_theta_lambda))
    return log_liklihood


if __name__ == '__main__':
    main()
