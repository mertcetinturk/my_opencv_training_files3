import os

import cv2
from os.path import isfile, join

pathIn = r'img1'
pathOut = 'MOT17-10-DPM.mp4'

files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

# img = cv2.imread(pathIn + '\\' + files[44])
# cv2.imshow('deneme', img)
# cv2.waitKey(0)

fps = 30
size = (1920, 1080)

# fourcc codecs
# fourcc = cv2.VideoWriter_fourcc(*'h264')
# or
# fourcc = cv2.VideoWriter_fourcc(*'x264')
# or
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(pathOut, fourcc, fps, size, True)

for i in files:
    print(i)

    fileName = pathIn + '\\' + i

    img = cv2.imread(fileName)

    out.write(img)

out.release()
cv2.destroyAllWindows()
