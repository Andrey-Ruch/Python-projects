import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt


def replacement_background(img, background, odir_name):
    background = cv2.imread(background)
    img = cv2.imread(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    fixed_image_result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(fixed_image_result)
    plt.show()

    fixed_image_result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    fixed_image_result = cv2.cvtColor(fixed_image_result, cv2.COLOR_BGR2RGB)
    plt.imshow(fixed_image_result)
    plt.show()

    # resize background
    if background.shape != img.shape:
        background = cv2.resize(background, (img.shape[1], img.shape[0]))

    # hsv green color
    green = np.uint8([[[0, 255, 0]]])
    hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
    print(hsv_green)

    # green limits
    lower_green = np.array([36, 60, 60])
    upper_green = np.array([80, 255, 255])

    # create masks
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)

    background_fix = cv2.bitwise_and(background, background, mask=mask)
    img_fix = cv2.bitwise_and(img, img, mask=mask_inv)

    # combining the image with the background
    result = cv2.add(img_fix, background_fix)

    cv2.imwrite(odir_name, result)


def main():
    img = sys.argv[1]
    background = sys.argv[2]
    odir_name = sys.argv[3]
    replacement_background(img, background, odir_name)


if __name__ == "__main__":
    main()
