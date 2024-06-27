from sklearn.impute import KNNImputer
from utils import *
import numpy as np
import matplotlib.pyplot as plt

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy (user), k={}: {}".format(k, acc))
    return acc, mat


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    transpose = np.transpose(matrix) # swap columns and rows
    mat = nbrs.fit_transform(transpose)
    mat = np.transpose(mat)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy (item), k={}: {}".format(k, acc))
    return acc, mat
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_vals = [1, 6, 11, 16, 21, 26]
    user_accs = []
    item_accs = []

    # (a) perform KNN for each k
    for k in k_vals:
        u_acc, mat = knn_impute_by_user(sparse_matrix, val_data, k)
        i_acc, mat = knn_impute_by_item(sparse_matrix, val_data, k)
        user_accs.append(u_acc)
        item_accs.append(i_acc)


    # (a) plot accuracy on validation
    plt.plot(k_vals, user_accs, label='User-based collaborative filtering')
    plt.plot(k_vals, item_accs, label='Item-based collaborative filtering')
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('Validation Data Accuracy v. K')
    plt.savefig('knn_valid_performance.png')



    # (b) determine optimal k*
    # ku* = 11, ki* = 22
    ku_star = k_vals[user_accs.index(max(user_accs))]
    ki_star = k_vals[item_accs.index(max(item_accs))]

    # calculate test accuracy for k*
    user_test_acc = knn_impute_by_user(sparse_matrix, test_data, ku_star)
    item_test_acc = knn_impute_by_item(sparse_matrix, test_data, ki_star)
    print("Test accuracy (user), k*={}: {}".format(ku_star, user_test_acc))
    print("Test accuracy (item), k*={}: {}".format(ki_star, item_test_acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

if __name__ == "__main__":
    main()
