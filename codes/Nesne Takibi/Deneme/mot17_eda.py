import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

col_list = ['frame_number', 'identity_number', 'left', 'top', 'width', 'height', 'score', 'class', 'visibility']

data = pd.read_csv('gt.txt', names=col_list)
pedestrian = data[data['class'] == 1]

videoPath = r'MOT17-10-DPM.mp4'
cap = cv2.VideoCapture(videoPath)

id1 = 2
number_of_image = np.max(data['frame_number'])
fps = 30
bound_box_list = []

for i in range(number_of_image-1):
    ret, frame = cap.read()

    if ret:
        frame = cv2.resize(frame, dsize=(960, 540))

        filter_id1 = np.logical_and(pedestrian['frame_number'] == i+1, pedestrian['identity_number'] == id1)

        if len(pedestrian[filter_id1]) != 0:
            x = int(pedestrian[filter_id1].left.values[0] / 2)
            y = int(pedestrian[filter_id1].top.values[0] / 2)
            w = int(pedestrian[filter_id1].width.values[0] / 2)
            h = int(pedestrian[filter_id1].height.values[0] / 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(frame, (int(x+w/2), int(y+h/2)), 2, (0, 0, 255), -1)

            # frame, x, y, genislik, yukseklik, center_x, center_y
            bound_box_list.append([i, x, y, w, h, int(x+w/2), int(y+h/2)])

        cv2.putText(frame, 'Frame Number' + str(i + 1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(15) == 27:
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(bound_box_list, columns=["frame_no", "x", "y", "w", "h", "center_x", "center_y"])
df.to_csv('gt_new.txt', index=False)
