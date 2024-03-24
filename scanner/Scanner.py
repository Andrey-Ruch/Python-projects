import cv2
import sys
import numpy as np


def get_the_biggest_contour(img_binary):
    (contours, _) = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_areas = []

    for contour in contours:
        area = cv2.contourArea(contour)
        contours_areas.append(area)

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return sorted_contours[0]


def get_the_parameters_of_bounding_box(img_copy, the_biggest_contour):
    # (3) Find the minimum area bounding box
    box = cv2.minAreaRect(the_biggest_contour)
    # (4) Find the coordinates of the four corners of the bounding box
    box_pts = np.int0(cv2.boxPoints(box))
    # (5) Calculation of height and width of the page
    height = ((((box_pts[0][0] - box_pts[3][0]) ** 2) + ((box_pts[0][1] - box_pts[3][1]) ** 2)) ** 0.5)
    width = ((((box_pts[1][0] - box_pts[0][0]) ** 2) + ((box_pts[1][1] - box_pts[0][1]) ** 2)) ** 0.5)

    if width > height:
        width, height = height, width
        return [box_pts[1], box_pts[2], box_pts[0], box_pts[3]], width, height

    return [box_pts[0], box_pts[1], box_pts[3], box_pts[2]], width, height


def Scanner(path_input_img, path_output_img):
    # Load the image
    img = cv2.imread(path_input_img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # (1) Binarize the image
    img_blurred = cv2.GaussianBlur(img_gray, (51, 51), 0)
    _, img_binary = cv2.threshold(img_blurred, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # (2) Find the biggest contour in the image
    img_copy = img.copy()
    the_biggest_contour = get_the_biggest_contour(img_binary)

    # (3) Find the minimum area bounding box
    # +
    # (4) Find the coordinates of the four corners of the bounding box
    # +
    # (5) Calculation of height and width of the page
    box_pts, width, height = get_the_parameters_of_bounding_box(img_copy, the_biggest_contour)

    # (6) The transformation matrix
    input_pts = np.float32(box_pts)
    output_pts = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # Create the transformation matrix M
    M = cv2.getPerspectiveTransform(input_pts, output_pts)

    # (7) Make a transformation
    result = cv2.warpPerspective(img_copy, M, (int(width), int(height)))

    # (8) Saving the output
    cv2.imwrite(path_output_img, result)


def main():
    path_input_img = sys.argv[1]
    path_output_img = sys.argv[2]
    Scanner(path_input_img, path_output_img)


if __name__ == "__main__":
    main()
