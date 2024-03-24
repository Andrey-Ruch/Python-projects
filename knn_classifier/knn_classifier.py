import cv2
import sys
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, confusion_matrix
import matplotlib.pyplot as plt


def padding(image_gray):
    height, width = image_gray.shape[0], image_gray.shape[1]
    padding_1 = padding_2 = int(0.5 * abs(height - width))

    if height > width:  # add padding in right and left
        image_gray = cv2.copyMakeBorder(image_gray, 0, 0, padding_1, padding_2, cv2.BORDER_CONSTANT, value=255)
    elif height < width:  # add padding in top and bottom
        image_gray = cv2.copyMakeBorder(image_gray, padding_1, padding_2, 0, 0, cv2.BORDER_CONSTANT, value=255)

    return image_gray


def pre_process(hhd_dataset_path):
    raw_images, labels = [], []

    for root, directories, files in os.walk(hhd_dataset_path):
        for name in files:
            image = cv2.imread(os.path.join(root, name))
            if image is None:
                print("Error with: " + str(root) + str(name))
                exit()

            # b. grayscale
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # c. padding
            image_gray = padding(image_gray)
            # d. resize
            image_gray = cv2.resize(image_gray, (32, 32))
            # e. binarization
            image_blurred = cv2.GaussianBlur(image_gray, (9, 9), 0)
            image_binary = cv2.threshold(image_blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            labels.append(name.split('_')[0])
            raw_images.append(image_binary.flatten())

    return raw_images, labels


def get_best_k(train_ri, val_ri, train_l, val_l, start, end, step):
    best_score, best_k, k = 0, 0, start

    while k <= end:
        knn = KNeighborsClassifier(n_neighbors=k)
        current_score = accuracy_score(val_l, knn.fit(train_ri, train_l).predict(val_ri))

        if best_score <= current_score:
            best_score = current_score
            best_k = k

        k += step

    return best_k


def creating_results_txt_file(best_k, test_l, predict_test):
    # Sorted by the order of the numbers of the letters
    array_of_matrixs = multilabel_confusion_matrix(test_l, predict_test)

    results_txt = open("results.txt", "w")
    results_txt.write("k = {0}\n".format(best_k))
    results_txt.write("Letter\t\tAccuracy\n")

    total_acc = 0

    for i in range(len(array_of_matrixs)):
        tp, fp = array_of_matrixs[i][1][1], array_of_matrixs[i][0][1]
        fn, tn = array_of_matrixs[i][1][0], array_of_matrixs[i][0][0]
        # accuracy = (tp + tn) / (tp + tn + fn + fp)
        accuracy = tp / (tp + fn)  # accuracy par letter of accuracy_score

        total_acc += accuracy

        results_txt.write("{0}\t\t{1:.2f}%\n".format(i, accuracy * 100))

    results_txt.close()


def creating_confusion_matrix_csv_file(test_l, predict_test):
    confusion_mtrx = confusion_matrix(test_l, predict_test).tolist()
    numeric_letters = ["Predicted / True label", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                       "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26"]

    for i in range(len(confusion_mtrx)):
        confusion_mtrx[i].insert(0, i)

    # create a new CSV file
    with open('confusion_matrix.csv', 'w', newline='') as csvfile:
        # create a CSV writer object
        writer = csv.writer(csvfile)

        writer.writerow(numeric_letters)
        writer.writerows(confusion_mtrx)


def knn_classifier(hhd_dataset_path):
    # 1. Pre-processing
    raw_images, labels = pre_process(hhd_dataset_path)

    # 2. Randomly dividing the dataset into train / val / test
    (trainRI, testRI, trainL, testL) = train_test_split(raw_images, labels, test_size=0.2, random_state=42)
    (valRI, testRI, valL, testL) = train_test_split(testRI, testL, test_size=0.5, random_state=42)

    # 3. Training
    best_k = get_best_k(trainRI, valRI, trainL, valL, 1, 15, 2)
    knn = KNeighborsClassifier(n_neighbors=best_k)

    # 4. Testing
    predict_test = knn.fit(trainRI, trainL).predict(testRI)

    # Results
    creating_results_txt_file(best_k, testL, predict_test)
    creating_confusion_matrix_csv_file(testL, predict_test)
    print('Accuracy score = ' + (str(accuracy_score(testL, predict_test))))


def main():
    hhd_dataset_path = sys.argv[1]
    knn_classifier(hhd_dataset_path)


if __name__ == "__main__":
    main()
