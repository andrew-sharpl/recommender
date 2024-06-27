import numpy as np
from part_b.data_cleaning import *
import pandas as pd


def create_theta_matrix(theta, omega, init='default'):
    n = len(theta)
    m = len(omega)
    theta_matrix = np.empty(shape=(n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            if init == 'random':
                theta_matrix[i, j] = np.random.rand()
            elif init == 'ones':
                theta_matrix[i, j] = 1
            else:
                theta_matrix[i, j] = theta[i] * (1 - omega[j])

    return theta_matrix


def create_lambda_matrix(data, omega, init='default'):
    n = len(set(data['question_id']))
    m = len(omega)
    lambda_matrix = np.empty(shape=(n, m), dtype=float)
    for i in range(n):
        if init == 'random':
            lambda_matrix[i] = [np.random.rand() for i in omega]
        elif init == 'ones':
            lambda_matrix[i] = [1 for i in omega]
        else:
            lambda_matrix[i] = omega

    return lambda_matrix


def create_beta_vector(data, meta, omega, init='default'):
    unique_questions = set(data['question_id'])
    beta_vector = np.ones((max(unique_questions) + 1,), dtype=float)
    for question_id in unique_questions:
        subject_difficulties = []
        question_subjects = list(meta.loc[meta['question_id'] == question_id]['subject_id'])[0]
        for subject in question_subjects:
            subject_difficulties.append(omega[subject])

        if init == 'random':
            beta_vector[question_id] = np.random.rand()
        elif init == 'ones':
            beta_vector[question_id] = 1
        else:
            beta_vector[question_id] = sum(subject_difficulties) / len(subject_difficulties)

    return beta_vector


def create_matrices(theta, omega, data, meta, init='default'):
    print("theta creation")
    theta_matrix = create_theta_matrix(theta, omega, init)
    print("lambda creation")
    lambda_matrix = create_lambda_matrix(data, omega, init)
    print("beta creation")
    beta_vector = create_beta_vector(data, meta, omega, init)

    return theta_matrix, lambda_matrix, beta_vector


if __name__ == '__main__':
    pass
