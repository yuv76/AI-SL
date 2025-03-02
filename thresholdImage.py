import cv2


def perform_threshold(image, thresh_val=141):
    # Grayscale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # add thresholding
    ret, thresh = cv2.threshold(gray_image, thresh_val, 255, cv2.THRESH_BINARY_INV)
    return thresh

