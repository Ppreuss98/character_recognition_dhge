import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def tess_setup():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


def show_image(image_name, image):
    cv2.imshow(image_name, image)
    cv2.waitKey(0)


def get_bounding_boxes(image):
    h, w, c = image.shape
    boxes = pytesseract.image_to_boxes(image)
    for b in boxes.splitlines():
        b = b.split(' ')
        image = cv2.rectangle(image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
    return image


def get_image_string(image):
    custom_config = r'-l deu --psm 6'
    string_image = pytesseract.image_to_string(image, config=custom_config)
    return string_image


def start():
    tess_setup()
    img = cv2.imread("image.jpg")
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    # show_image("Gray image", get_grayscale(img))
    # show_image("Noise removed", remove_noise(img))
    # show_image("Eroded", erode(img))
    # show_image("Opening", opening(img))
    # show_image("Canny", canny(img))
    # show_image("BB", get_bounding_boxes(img))

    print(get_image_string(img))


start()
