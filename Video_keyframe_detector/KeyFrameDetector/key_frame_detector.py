import os
import cv2
import csv
import numpy as np
import time
import peakutils
from Video_keyframe_detector.KeyFrameDetector.utils import convert_frame_to_grayscale, prepare_dirs, plot_metrics


def keyframeDetection(source, dest, Thres, plotMetrics=False, verbose=False):
    keyframePath = dest + '/keyFrames'
    imageGridsPath = dest + '/imageGrids'
    csvPath = dest + '/csvFile'
    path2file = csvPath + '/output.csv'
    prepare_dirs(keyframePath, imageGridsPath, csvPath)

    cap = cv2.VideoCapture(source)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if (cap.isOpened() == False):
        print("Error opening video file")

    lstfrm = []
    lstdiffMag = []
    timeSpans = []
    images = []
    full_color = []
    lastFrame = None
    Start_time = time.process_time()

    # Read until video is completed
    for i in range(length):
        ret, frame = cap.read()
        grayframe, blur_gray = convert_frame_to_grayscale(frame)

        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
        lstfrm.append(frame_number)
        images.append(grayframe)
        full_color.append(frame)
        if frame_number == 0:
            lastFrame = blur_gray
        # subtracting current frame to previous frame to find different pixel magnitude between two frames
        diff = cv2.subtract(blur_gray, lastFrame)
        # count nonezero pixel of the array
        diffMag = cv2.countNonZero(diff)
        lstdiffMag.append(diffMag)
        stop_time = time.process_time()
        time_Span = stop_time - Start_time
        timeSpans.append(time_Span)
        lastFrame = blur_gray
    cap.release()
    y = np.array(lstdiffMag)
    # remove small value by using polynomial equation....
    base = peakutils.baseline(y, 2)
    # filter based on Thres
    indices = peakutils.indexes(y - base, Thres, min_dist=1)

    ##plot to monitor the selected keyframe
    if (plotMetrics):
        plot_metrics(indices, lstfrm, lstdiffMag)

    cnt = 1
    for x in indices:
        cv2.imwrite(os.path.join(keyframePath, str(cnt) + '.jpg'), full_color[x])
        cnt += 1
        log_message = str(cnt) + ' happened at ' + str(timeSpans[x]) + ' sec.'
        if (verbose):
            print(log_message)
        with open(path2file, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(log_message)
            csvFile.close()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    source = "/home/tuan/Documents/Code/mmaction2/demo/practical_video/putting_on_2.mp4"
    dest = "/home/tuan/Downloads"
    Thres = 0.3
    keyframeDetection(source=source, dest=dest, Thres=Thres)

