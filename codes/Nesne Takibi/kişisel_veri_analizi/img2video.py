"""
resim2video
"""
import os

import cv2
from os.path import isfile, join
import matplotlib.pyplot as plt

pathIn = r"img1"
pathOut = "deneme.mp4"

files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

# img = cv2.imread(pathIn + '\\' + files[44])
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img), plt.axis('off'), plt.show()

fps = 25
size = (1920, 1080)

# fourcc codecs
# fourcc = cv2.VideoWriter_fourcc(*'h264')
# or
# fourcc = cv2.VideoWriter_fourcc(*'x264')
# or
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(pathOut, fourcc, fps, size, True)
# fourcc codeclerinin hepsi çalışıyor ancak cv2.VideoWriter'ın içine yazınca hata aldım
# fourcc codeclerini ayrı bir obje olarak tanımlayıp cv2.VideoWriter'ın içine öyle yaz.

for i in files:
    print(i)

    filename = pathIn + "\\" + i

    img = cv2.imread(filename)

    out.write(img)

out.release()
cv2.destroyAllWindows()
