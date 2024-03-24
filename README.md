# Image Processing and Computer Vision projects

* [Gender Classification from Handwriting](#Gender-Classification-from-Handwriting)
* [Classifier for Handwritten Hebrew Letters](#Classifier-for-Handwritten-Hebrew-Letters)
* [Background replacement](#Background-replacement)
* [Scanner](#Scanner)

## Gender Classification from Handwriting
Classifying a manuscript according to the gender of the writer.
Automatic classification of the writer's gender is an essential task in a wide variety of fields, for example, psychology, classification of historical documents or criminological analysis. Psychological studies of handwriting analysis have confirmed that gender classification can be made according to a number of significant differences in handwriting. In general, while a woman's handwriting tends to be more uniform, neat and circular, a man's handwriting tends to be more pointed, messy and slanted.

The purpose of the project is to train an SVM model for the automatic classification of a handwriting image according to the gender of the writer.
To do this, we will perform experiments with different parameters and kernels, and report which combination of parameters achieves the highest accuracy

The HDD_gender dataset was used to train the model. This dataset contains around 850 samples of Hebrew handwriting together with their labels.

### Samples from the gender_HHD dataset
 | ![HDD_gender_example_1](https://github.com/Andrey-Ruch/Python-projects/assets/73066767/b7455ec4-eb8f-4101-b474-fc7b6eae413b?raw=true) |
 |:--:| 
 | *Left - A man's handwriting / Right - A woman's handwriting* |

### Result
    Values of the parameters that give the highest accuracy:
    - Radius = 3
    - Number of points = 24
    - kernel = linear, with parameters: SVC(kernel='linear')

    Accuracy: 81.43%

    Confusion matrix:
            |  male  |  female  
    ----------------------------
      male  |   29   |    6     
    ----------------------------
     female |   7    |    28     

 ### Getting Started
 1) Before running the code:
    * Make sure you have the necessary images folders from gender_HHD dataset [1]
    * Make sure you have python version 3.10.8
    * Install numpy using pip install numpy
    * Install sklearn using pip install sklearn
    * Install skimage using pip install skimage
    * Make sure that you are in the folder where the classifier.py is

2) To run this program, enter the following into the command prompt:

    * python classifier.py <path_of_train_images_folder> <path_of_val_folder> <path_of_test_folder>

 ### References
 [1] [Automatic Gender Classification from Handwritten Images: A Case Study](https://link.springer.com/chapter/10.1007/978-3-030-89131-2_30)

## Classifier for Handwritten Hebrew Letters
The purpose of the project is to classify images of letters from the HHD_0 dataset, which consists of handwritten letters in Hebrew.
For this we will train the KNN classifier for letter classificationץ

The HHD_v0 dataset contains around 5000 images of individual letters. These images are divided into 27 subgroups (subfolders). Each folder contains images of a certain letter from the Hebrew alphabet. [Details about the dataset](https://www.researchgate.net/publication/343880780_The_HHD_Dataset).

 | ![HHD_v0_example](https://github.com/Andrey-Ruch/Python-projects/assets/73066767/afc7686b-2f5d-4a3e-be3d-0bd2ed4058a0) |
 |:--:| 
 | *A sample from the HHD_v0 dataset of handwritten letters* |

 ### Result
 The k value that gives the highest accuracy and Accuracy reached by the classifier for each of the letters (27 different letters):

    k = 7
    Letter		Accuracy
    0		72.00%
    1		60.00%
    2		88.00%
    3		61.54%
    4		80.95%
    5		90.00%
    6		64.29%
    7		82.35%
    8		93.75%
    9		100.00%
    10		66.67%
    11		75.00%
    12		92.31%
    13		29.17%
    14		68.42%
    15		28.57%
    16		78.95%
    17		84.21%
    18		73.68%
    19		76.47%
    20		52.17%
    21		37.50%
    22		60.00%
    23		86.67%
    24		76.47%
    25		73.68%
    26		84.21%

 ### Getting Started
 1) Before running the code:
    * Make sure you have the necessary images folder from HHD_v0 dataset [2]
    * Make sure you have python version 3.10.8
    * Install numpy using pip install numpy
    * Install numpy using pip install sklearn
    * Make sure that you are in the folder where the knn_classifier.py is

2) To run this program, enter the following into the command prompt:

    * python knn_classifier.py <path_of_images_folder>

 ### References
 [2] [The HHD Dataset](https://www.researchgate.net/publication/343880780_The_HHD_Dataset)

 ## Background replacement
 The purpose of the project is to change the background for a photo taken with a green screen.

| ![Replacer_example](https://github.com/Andrey-Ruch/Python-projects/assets/73066767/cdeaf16a-ae78-469b-beba-8b2ba597cd74) |
|:--:|
| *Left - Original picture / Right - Result* |

 ### Getting Started
 1) Before running the code:
    * Make sure you have the necessary images folder
    * Make sure you have python version 3.10.8
    * Install openCV using pip install opencv-contrib-python
    * Install numpy using pip install numpy
    * Make sure that you are in the folder where the replacer.py is

2) To run this program, enter the following into the command prompt:

    * python replacer.py .\images\\<picture-name-with-a-green-screen>.jpg .\images\\<background-picture-name>.jpg <out-put-picture-name>.jpg

## Scanner
The purpose of the project is to take a photo of a rectangular page (like an A4 page or a photo of a book) and align the photo - similar to how the CamScanner application works.

| ![Scanner_example](https://github.com/Andrey-Ruch/Python-projects/assets/73066767/ac004267-6823-49c1-9b91-09130d79adc8) |
|:--:|
| *Left - Original picture / Right - Result* |

 ### Getting Started
 1) Before running the code:
    * Make sure you have the necessary images in picture folder (input)
    * Make sure you have python version 3.10.8
    * Install openCV using pip install opencv-contrib-python
    * Install numpy using pip install numpy
    * Make sure that you are in the folder where the Scanner.py is

2) To run this program, enter the following into the command prompt:

    * python Scanner.py .\input\\<input_img>.jpg <path_output_img>.jpg
