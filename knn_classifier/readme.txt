* Description
k-Nearest Neighbor Algorithm for classifying images of letters from the HHD_0 dataset,
which consists of handwritten letters in Hebrew.

* Environment
* Describe the OS and compilation requirements needed to compile and run the program
* How to Run Your Program
1) Before running the code:
    • Make sure you have the necessary images in picture folder (hhd_dataset) [1].
    • Make sure you have python version 3.10.8
    • Install openCV using pip install opencv-contrib-python.
    • Install numpy using pip install numpy.
    • Install numpy using pip install sklearn.
    • Make sure that you are in the folder where the knn_classifier.py is

2) To run this program, enter the following into the command prompt:

    python knn_classifier.py <path_of_images_dataset>

* Output
    • result.txt - Contains accuracy reached by the classifier for each of the letters (27 different letters).
    • confusion_matrix.csv

* References
    [1] https://www.researchgate.net/publication/343880780_The_HHD_Dataset