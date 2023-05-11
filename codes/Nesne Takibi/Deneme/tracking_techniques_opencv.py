import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

OPENCV_OBJECT_TRACKERS = {'csrt'       : cv2.legacy.TrackerCSRT_create,
                          'kcf'        : cv2.legacy.TrackerKCF_create,
                          'boosting'   : cv2.legacy.TrackerBoosting_create,
                          'mil'        : cv2.TrackerMIL_create,
                          'tld'        : cv2.legacy.TrackerTLD_create,
                          'medianflow' : cv2.legacy.TrackerMedianFlow_create,
                          'mosse'      : cv2.legacy.TrackerMOSSE_create}

tracker_name = 'csrt'
tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()
print(f'Tracker Name = {tracker_name}')

gt = pd.read_csv('gt_new.txt')
video_path = r'MOT17-10-DPM.mp4'
cap = cv2.VideoCapture(video_path)

# genel parametreler
initBB = None
fps = 30
frame_number = []
f = 0
success_track_frame_success = 0
track_list = []
start_time = time.time()  # geçen süreye bakmak için

while True:

    ret, frame = cap.read()

    if ret:
        frame = cv2.resize(frame, dsize=(960, 540))
        (H, W) = frame.shape[:2]

        pedestrian_gt = gt[gt.frame_no == f]
        if len(pedestrian_gt) != 0:
            x = pedestrian_gt.x.values[0]
            y = pedestrian_gt.y.values[0]
            w = pedestrian_gt.w.values[0]
            h = pedestrian_gt.h.values[0]

            center_x = pedestrian_gt.center_x.values[0]
            center_y = pedestrian_gt.center_y.values[0]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.circle(frame, (center_x, center_y), 2, (255, 0, 0), cv2.FILLED)

        # box
        if initBB is not None:
            (success, box) = tracker.update(frame)

            if f <= np.max(gt.frame_no):
                (x, y, w, h) = [int(i) for i in box]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                success_track_frame_success += 1
                tracker_center_x = int(x + w / 2)
                tracker_center_y = int(y + h / 2)
                track_list.append([f, tracker_center_x, tracker_center_y])

            info = [('Tracker', tracker_name),
                    ('Success', 'Yes' if success else 'No')]

            for (i , (o, p)) in enumerate(info):
                text = f'{o}: {p}'
                cv2.putText(frame, text, (10, H - (i * 20) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame, 'Frame Number: ' + str(f), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)

        if key == 27:
            break

        if key == ord('t'):
            initBB = cv2.selectROI('Frame', frame, fromCenter=False)
            tracker.init(frame, initBB)

        # frame
        frame_number.append(f)
        f += 1

    else:
        break
cap.release()
cv2.destroyAllWindows()

stop_time = time.time()
time_diff = stop_time - start_time

track_df = pd.DataFrame(track_list, columns=['frame_no', 'center_x', 'center_y'])

if len(track_df) is not None:
    print(f'Tracker Method: {tracker}')
    print(f'Time: {time_diff}')
    print(f'Number of Frame to Track (gt): {len(gt)}')
    print(f'Number of Frame to Track (track success): {success_track_frame_success}')

    track_df_frame = track_df.frame_no

    gt_center_x = gt.center_x[track_df_frame].values
    gt_center_y = gt.center_y[track_df_frame].values

    track_df_center_x = track_df.center_x.values
    track_df_center_y = track_df.center_y.values

    plt.plot(np.sqrt((gt_center_x - track_df_center_x) ** 2 + (gt_center_y - track_df_center_y) ** 2))
    plt.xlabel('Frame')
    plt.ylabel('Euclidean Distance')
    error = np.sum(np.sqrt((gt_center_x - track_df_center_x) ** 2 + (gt_center_y - track_df_center_y) ** 2))
    print('Toplam Hata: ', error)
