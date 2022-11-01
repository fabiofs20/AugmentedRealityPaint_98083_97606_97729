#!/usr/bin/env python

# ---------- Augmented Reality Paint ------------
# Filipe Goncalves, 98083
# Diogo Monteiro, 97606
# Fábio Silva, 97729
# 
# PSR, October 2022.
# -----------------------------------------------

from calendar import c
import cv2
from functools import partial
import numpy as np
import copy
import json
from colorama import Fore, Back, Style

def processImage(ranges, image):

    # processing
    mins = np.array([ranges['B']['min'], ranges['G']['min'], ranges['R']['min']])
    maxs = np.array([ranges['B']['max'], ranges['G']['max'], ranges['R']['max']])

    # mask
    mask = cv2.inRange(image, mins, maxs)
    # conversion from numpy from uint8 to bool
    mask = mask.astype(bool)

    # process the image
    image_processed = copy.deepcopy(image)
    image_processed[np.logical_not(mask)] = 0

    # get binary image with threshold the values not in the mask
    _, image_processed = cv2.threshold(image_processed, 1, 255, cv2.THRESH_BINARY)

    return image_processed

def onTrackbar(value, channel, min_max, ranges):
    print("Selected threshold "+ Style.BRIGHT + Fore.YELLOW + str(value) + Style.RESET_ALL + " for limit " + Style.BRIGHT + Fore.GREEN + channel + min_max + Style.RESET_ALL)

    # update range values
    ranges[channel][min_max] = value


def main():

    # start video capture
    capture = cv2.VideoCapture(0)

    # windows
    window_name_seg = 'Segmented'
    window_name_or = 'Original'
    cv2.namedWindow(window_name_seg,cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(window_name_or,cv2.WINDOW_AUTOSIZE)

    # range limits
    ranges = {'B': {'max': 200, 'min': 100},
              'G': {'max': 200, 'min': 100},
              'R': {'max': 200, 'min': 100}}

    # different trackbars for each min and max value for each channel
    cv2.createTrackbar('Rmin', window_name_seg, ranges["R"]["min"], 255, partial(onTrackbar, channel='R', min_max='min', ranges=ranges))
    cv2.createTrackbar('Rmax', window_name_seg, ranges["R"]["max"], 255, partial(onTrackbar, channel='R', min_max='max', ranges=ranges))
    cv2.createTrackbar('Gmin', window_name_seg, ranges["G"]["min"], 255, partial(onTrackbar, channel='G', min_max='min', ranges=ranges))
    cv2.createTrackbar('Gmax', window_name_seg, ranges["G"]["max"], 255, partial(onTrackbar, channel='G', min_max='max', ranges=ranges))
    cv2.createTrackbar('Bmin', window_name_seg, ranges["B"]["min"], 255, partial(onTrackbar, channel='B', min_max='min', ranges=ranges))
    cv2.createTrackbar('Bmax', window_name_seg, ranges["B"]["max"], 255, partial(onTrackbar, channel='B', min_max='max', ranges=ranges))

    # while user wants video capture
    while True:

        # get frame
        ret, image = capture.read()

        # get key
        k = cv2.waitKey(1)

        # error getting the frame
        if not ret:
            print(Style.BRIGHT + Fore.RED + "Failed to grab frame" + Style.RESET_ALL)
            break

        # process image
        processed_image = processImage(ranges, image)
        cv2.imshow(window_name_seg, processed_image)

        cv2.imshow(window_name_or, image)

        # user quits
        if k == ord("q"):
            break

        # writes range limits to file
        if k == ord("w"):
            file_name = 'limits.json'
            with open(file_name, 'w') as file_handle:
                print('writing color limits to file ' + Style.BRIGHT + Fore.GREEN + file_name + Style.RESET_ALL)
                json.dump({"limits": ranges}, file_handle)
            print({"limits": ranges})
            break

    # end capture and destroy windows
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()