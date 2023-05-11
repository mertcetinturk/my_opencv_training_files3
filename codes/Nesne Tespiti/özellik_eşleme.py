import cv2
import matplotlib.pyplot as plt

# ana görüntüyü içe aktar
chos = cv2.imread(r'../../images/chocolates.jpg', 0)

cho = cv2.imread(r'../../images/nestle.jpg', 0)

# orb tanımlayıcısı
# köşe, kenar gib nesneye ait özellikler

orb = cv2.ORB_create()

# anahtar nokta tespiti
kp1, des1 = orb.detectAndCompute(cho, None)
kp2, des2 = orb.detectAndCompute(chos, None)

# brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# noktaları eşleştir
matches = bf.match(des1, des2)

# mesafeye göre sırala
matches = sorted(matches, key=lambda x: x.distance)

# eşleşen nesneleri görselleştirelim
plt.figure()
img_match = cv2.drawMatches(cho, kp1, chos, kp2, matches[:20], None, flags=2)
plt.imshow(img_match), plt.axis('off')
plt.show()

# sift yöntemi kullanılarak daha iyi sonuçlar elde edilebilir
# sift opencv ye sonradan eklenmiş bir özelliktir, bu yüzden install yapmak gerekir
# yapmadığımız için zip dosyası içinden feature_matching.py ile çalışmak daha doğru olacaktır.
