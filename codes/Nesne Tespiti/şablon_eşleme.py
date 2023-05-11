import cv2
import matplotlib.pyplot as plt

# template matching
img = cv2.imread(r'../../images/cat.jpg', 0)
template = cv2.imread(r'../../images/cat_face.jpg', 0)

h, w = template.shape

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:

    method = eval(meth)  # string halinden fonksiyon haline dönüştürüyor

    res = cv2.matchTemplate(img, template, method)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc

    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img, top_left, bottom_right, (255, 0, 255), 2)

    plt.figure()
    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Eşleşen Sonuç'), plt.axis('off')

    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Tespit Edilen Sonuç'), plt.axis('off')
    plt.suptitle(meth)
    plt.show()
