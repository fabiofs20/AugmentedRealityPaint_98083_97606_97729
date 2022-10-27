#!/usr/bin/env python

# ---------- Augmented Reality Paint ------------
# Filipe Goncalves, 98083
# Diogo Monteiro, 97606
# Fábio Silva, 97729
# 
# PSR, October 2022.
# -----------------------------------------------

import argparse
import cv2
import numpy as np
import copy
import json
import time

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

    return image_processed, mask

def main():

    parser = argparse.ArgumentParser(description='Definition of test mode')
    parser.add_argument('-j', '--json', required=True, dest='JSON', help='Full path to json file.')
    parser.add_argument('-v', '--video-canvas', required=False, help='Use video streaming as canvas', action="store_true", default=False)
    parser.add_argument('-p', '--paint-numeric', required=False, help='Use a numerical canvas to paint', action="store_true", default=False)
    parser.add_argument('-s', '--shake-detection', required=False, help='Use shake detection', action="store_true", default=False)

    args = vars(parser.parse_args())
    print(args)

    # start video capture
    capture = cv2.VideoCapture(0)

    # windows
    window_name = 'Original'
    cv2.namedWindow(window_name,cv2.WINDOW_AUTOSIZE)
    window_name_paint = 'Painter'
    cv2.namedWindow(window_name_paint,cv2.WINDOW_AUTOSIZE)
    window_name_segmented = 'Segmented'
    cv2.namedWindow(window_name_segmented,cv2.WINDOW_AUTOSIZE)
    window_name_area = 'Largest Area'
    cv2.namedWindow(window_name_area,cv2.WINDOW_AUTOSIZE)

    # Opening JSON file
    f = open(args["JSON"])
    data = json.load(f)

    # painter variables
    _, frame = capture.read()

    painter = np.ones((frame.shape[0], frame.shape[1], 3)) * 255
    color = (0,0,0)
    size_brush = 5
    last_point = None

    mr = np.ones((frame.shape[0], frame.shape[1], 3)) * 0

    # while user wants video capture
    while True:

        # get frame
        ret, image = capture.read()

        h, w = image.shape[0:2]

        # get key
        k = cv2.waitKey(1)

        # error getting the frame
        if not ret:
            print("failed to grab frame")
            break

        image_p, mask = processImage(data["limits"], image)

        connectivity = 4  
        # Perform the operation
        nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(cv2.cvtColor(image_p, cv2.COLOR_BGR2GRAY), connectivity, cv2.CV_32S)
        # Find the largest non background component.
        # Note: range() starts from 1 since 0 is the background label.
        if nb_components > 1:
            max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])

            mr = np.equal(labels, max_label)

            b,g,r = cv2.split(image)
            b[mr] = 0
            r[mr] = 0
            g[mr] = 200
            image = cv2.merge((b,g,r))

            # put text and highlight the center
            cv2.line(image, (int(centroids[max_label][0])+5, int(centroids[max_label][1])), (int(centroids[max_label][0])-5, int(centroids[max_label][1])), (0,0,255), 5, -1)
            cv2.line(image, (int(centroids[max_label][0]), int(centroids[max_label][1])+5), (int(centroids[max_label][0]), int(centroids[max_label][1])-5), (0,0,255), 5, -1)
            if last_point != None:
                cv2.line(painter, last_point, (int(centroids[max_label][0]), int(centroids[max_label][1])), color, size_brush, -1)
            last_point = (int(centroids[max_label][0]), int(centroids[max_label][1]))
        else:
            print("Please place your object in front of the camera")

        if args["video_canvas"]:
            mask = np.not_equal(cv2.cvtColor(painter, cv2.COLOR_BGR2GRAY), 255)
            image[mask] = painter
            painter = image

        cv2.imshow(window_name, image)
        cv2.imshow(window_name_paint, painter)
        cv2.imshow(window_name_segmented, image_p)
        cv2.imshow(window_name_area, mr.astype(np.uint8)*255)

        # user quits
        if k == ord("q"):
            break

        # brush
        if k == ord("+"):
            size_brush += 1
        if k == ord("-"):
            size_brush = max(2, size_brush-1)

        # color
        if k == ord("r"):
            color = (0,0,255)
        if k == ord("g"):
            color = (0,255,0)
        if k == ord("b"):
            color = (255,0,0)

        # clear
        if k == ord("c"):
            painter = np.ones((600, 600, 3)) * 255
        
        # save
        if k == ord("w"):
            cv2.imwrite(f"drawing_{(time.ctime(time.time())).replace(' ', '_')}.png", painter)


    # end capture and destroy windows
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()