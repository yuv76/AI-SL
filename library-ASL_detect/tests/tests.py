# tests/test_utils.py
import unittest
import cv2
from ASL_detect import cnn_single_image, r_cnn_single_image

class TestASL(unittest.TestCase):
    def test_cnn(self):
        a_path = "A.png"
        f_path = "F.png"
        r_path = "R.png"
        l_path = "L.png"

        # passing image 1 - A
        image1 = cv2.imread(a_path)
        guess = cnn_single_image(image1, -1)
        print(f"IMAGE 1:\nExpected: A\nGot: {guess[0]} in {guess[1]*100}% certainity")

        # passing image 2 - F
        image2 = cv2.imread(f_path)
        guess = cnn_single_image(image2, -1)
        print(f"IMAGE 2:\nExpected: F\nGot: {guess[0]} in {guess[1]*100}% certainity")

        # passing image 3 - R
        image3 = cv2.imread(r_path)
        guess = cnn_single_image(image3, -1)
        print(f"IMAGE 3:\nExpected: R\nGot: {guess[0]} in {guess[1]*100}% certainity")

        # passing image 4 - not thresholded - L
        image4 = cv2.imread(l_path)
        guess = cnn_single_image(image4)
        print(f"IMAGE 3:\nExpected: L\nGot: {guess[0]} in {guess[1]*100}% certainity")


    def test_rcnn(self):
        E_path = "E_whole.png"
        W_path = "W_whole.png"

        # passing image 1 - E
        image1 = cv2.imread(E_path)
        guess = r_cnn_single_image(image1)
        print(f"IMAGE 1:\nExpected: E\nGot: {guess}")

        # passing image 2 - W
        image2 = cv2.imread(W_path)
        guess = r_cnn_single_image(image2)
        print(f"IMAGE 1:\nExpected: W\nGot: {guess}")


if __name__ == "__main__":
    unittest.main()
