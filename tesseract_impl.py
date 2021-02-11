import cv2
import pytesseract


def tess_setup():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def alter_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, 150, 200)
    return img


def show_image(img_name, img):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)


def get_image_string(img):
    custom_config = r'-l deu'
    string_image = pytesseract.image_to_string(img, config=custom_config)
    return string_image


def start():
    tess_setup()
    img = cv2.imread("image.jpg")
    img = alter_image(img)
    print(get_image_string(img))


start()
