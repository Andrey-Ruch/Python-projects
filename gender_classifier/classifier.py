import sys
import os
import cv2
import numpy as np
from sklearn import svm
from skimage import feature
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import time


start_time = time.time()


def lbp_feature(images, radius, num_of_points):
    print('• lbp_feature: {:.2f} minutes'.format((time.time() - start_time) / 60))
    lbp_features = []

    for image in images:
        lpb = feature.local_binary_pattern(image, num_of_points, radius, method="uniform")
        (hist, _) = np.histogram(lpb.ravel(), bins=range(0, num_of_points + 3), range=(0, num_of_points + 2))
        lbp_features.append(hist)

    return lbp_features


def load_images_dataset(path):
    print('• load_images_dataset: {:.2f} minutes'.format((time.time() - start_time) / 60))
    images, labels = [], []

    for sub_folder in ['female', 'male']:
        path_ = os.path.join(path, sub_folder)
        for file_name in os.listdir(path_):
            image = cv2.imread(os.path.join(path_, file_name))
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_blurred = cv2.GaussianBlur(image_gray, (15, 15), 0)

            labels.append(sub_folder)
            images.append(image_blurred)

    return images, labels


def get_the_best_accuracy(train_features, train_labels, val_features, val_labels):
    print('• get_the_best_accuracy: {:.2f} minutes'.format((time.time() - start_time) / 60))
    # Parameter Grid
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}

    # Create a linear SVM classifier
    clf_linear = svm.SVC(kernel='linear')
    clf_linear.fit(train_features, train_labels)
    linear_accuracy = clf_linear.score(val_features, val_labels)

    # Make grid search classifier
    clf_grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, verbose=1)
    clf_grid.fit(train_features, train_labels)
    clf_rbf = clf_grid.best_estimator_
    rbf_accuracy = clf_rbf.score(val_features, val_labels)

    # Choose the best model based on the highest accuracy on the validation set
    if linear_accuracy > rbf_accuracy:
        print('• linear is better')
        print('  with accuracy: ' + str(linear_accuracy))
        print('--------------------------------------')
        clf, best_accuracy = clf_linear, linear_accuracy
    else:
        print('• rbf is better')
        print(clf_grid.best_estimator_)
        print('  with accuracy: ' + str(rbf_accuracy))
        print('--------------------------------------')
        clf, best_accuracy = clf_rbf, rbf_accuracy

    return clf, best_accuracy


def creating_results_txt_file(best_parameters, best_clf, accuracy, conf_mtrx):
    print('• Creating_results_txt_file: {:.2f} minutes'.format((time.time() - start_time) / 60))

    with open("results.txt", "w") as file:
        file.write('Values of the parameters that give the highest accuracy:\n')
        file.write('- Radius = {}\n- Number of points = {}\n'.format(best_parameters[0], best_parameters[1]))
        file.write('- kernel = ' + str(best_clf.kernel) + ', with parameters: ' + str(best_clf))
        file.write('\n\nAccuracy: {:.2f}%\n\n'.format(accuracy * 100))
        file.write('Confusion matrix:')
        file.write('\n        |  male  |  female  \n')
        file.write('----------------------------\n')
        file.write('  male  |   {}   |    {}     \n'.format(conf_mtrx[0][0], conf_mtrx[0][1]))
        file.write('----------------------------\n')
        file.write(' female |   {}   |    {}     \n'.format(conf_mtrx[1][0], conf_mtrx[1][1]))

    print('• The results.txt file is ready.')


def classifier(path_train, path_val, path_test):
    # (1) Load images dataset
    train_images, train_labels = load_images_dataset(path_train)
    val_images, val_labels = load_images_dataset(path_val)
    test_images, test_labels = load_images_dataset(path_test)

    # Parameters of radius and number of points for LBP features
    best_parameters, best_accuracy, best_clf = [], 0, None
    lbp_parameters = ([1, 8], [3, 24])

    for radius, num_of_points in lbp_parameters:
        print('• Feature extraction with, radius: ' + str(radius) + ' num_of_points: ' + str(num_of_points))
        # (2) LBP feature
        train_features = lbp_feature(train_images, radius, num_of_points)
        val_features = lbp_feature(val_images, radius, num_of_points)

        # (3) Training
        clf, temp_accuracy = get_the_best_accuracy(train_features, train_labels, val_features, val_labels)

        if best_accuracy <= temp_accuracy:
            best_clf, best_accuracy = clf, temp_accuracy
            best_parameters = [radius, num_of_points]

    # (4) Evaluation of the SVM on the test set
    print('• Evaluation of the SVM on the test set')
    test_features = lbp_feature(test_images, best_parameters[0], best_parameters[1])
    accuracy = best_clf.score(test_features, test_labels)
    conf_mtrx = confusion_matrix(test_labels, best_clf.predict(test_features), labels=['male', 'female'])

    # Results
    creating_results_txt_file(best_parameters, best_clf, accuracy, conf_mtrx)


def main():
    print('• Start time: ', start_time)

    path_train, path_val, path_test = sys.argv[1], sys.argv[2], sys.argv[3]

    classifier(path_train, path_val, path_test)

    end_time = time.time()
    total_time = end_time - start_time
    print("• Time taken: {:.2f} minutes".format(total_time / 60))


if __name__ == "__main__":
    main()
