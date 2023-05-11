import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread(r'../../images/london.jpg', 0)

plt.figure(), plt.imshow(img, cmap='gray'), plt.axis('off')

edges = cv2.Canny(image=img, threshold1=0, threshold2=255)
plt.figure(), plt.imshow(edges, cmap='gray'), plt.axis('off')

median_val = np.median(img)
print(f'Median Val: {median_val}')
mean_val = np.mean(img)
print(f'Mean Val: {mean_val}')

low = int(max(0, (1 - 0.33) * median_val))  # alt threshold değeri
high = int(min(255, (1 + 0.33) * median_val))  # üst threshol değeri

print(f'Low: {low}'), print(f'High: {high}')

edges = cv2.Canny(image=img, threshold1=low, threshold2=high)
plt.figure(), plt.imshow(edges, cmap='gray'), plt.axis('off')

# blur

blurred_img = cv2.blur(img, ksize=(5, 5))
plt.figure(), plt.imshow(blurred_img, cmap='gray'), plt.axis('off')

median_val = np.median(blurred_img)

low2 = int(max(0, (1 - 0.33) * median_val))  # alt threshold değeri
high2 = int(min(255, (1 + 0.33) * median_val))  # üst threshol değeri


edges2 = cv2.Canny(image=blurred_img, threshold1=low2, threshold2=high2)
plt.figure(), plt.imshow(edges2, cmap='gray'), plt.axis('off')

plt.show()
