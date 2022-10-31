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
    parser.add_argument('-s', '--use-shake-detection', required=False, help='Use shake detection', action="store_true", default=False)
    parser.add_argument('-m', '--use-mouse', required=False, help='Use mouse as brush instead of centroid', action="store_true", default=False)

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
    height, width = frame.shape[0:2]

    painter = np.ones((frame.shape[0], frame.shape[1], 3), np.uint8) * 255
    color = (0,0,0)
    size_brush = 5
    last_point = None
    # min distance_squared between two consecutive points to be detected as an oscilation
    shake_detection_threshold = 1600

    mr = np.zeros((frame.shape[0], frame.shape[1], 3))


    mouse_coords = {'x': None, 'y': None}

    if args['use_mouse']:
        # pass {'x': int, 'y': int} dict as param
        def mouseHoverCallback(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                # mirror x coordinate, as the drawing is flipped horizontally
                param['x'] = width - int(x)
                param['y'] = int(y)
        cv2.setMouseCallback(window_name_paint, mouseHoverCallback, mouse_coords)


    # while user wants video capture
    while True:

        # get frame
        ret, image = capture.read()
        cam_output = image

        # get key
        k = cv2.waitKey(1)

        # error getting the frame
        if not ret:
            print("failed to grab frame")
            break

        image_p, mask = processImage(data["limits"], image)

        centroid = None

        # use mouse as brush
        if args['use_mouse'] and mouse_coords['x'] is not None:
            centroid = (mouse_coords['x'], mouse_coords['y'])


        connectivity = 4  
        # Perform the operation
        nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(cv2.cvtColor(image_p, cv2.COLOR_BGR2GRAY), connectivity, cv2.CV_32S)
        # Find the largest non background component.
        # Note: range() starts from 1 since 0 is the background label.
        if nb_components > 1:
            max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])

            centroid = (int(centroids[max_label][0]), int(centroids[max_label][1]))

            # highlight largest area
            mr = np.equal(labels, max_label)
            b,g,r = cv2.split(image)
            b[mr] = 0
            r[mr] = 0
            g[mr] = 200
            cam_output = cv2.merge((b,g,r))

            # put text and highlight the center
            cv2.line(cam_output, (centroid[0]+5, centroid[1]), (centroid[0]-5, centroid[1]), (0,0,255), 5, -1)
            cv2.line(cam_output, (centroid[0], centroid[1]+5), (centroid[0], centroid[1]-5), (0,0,255), 5, -1)
        else:
            mr = np.zeros((frame.shape[0], frame.shape[1], 3))
            #print("Please place your object in front of the camera")

        if last_point is not None and centroid is not None:
            # get squared distance between current point and previous
            distance = (last_point[0]-centroid[0])**2 + (last_point[1]-centroid[1])**2

            # oscilation detected, draw a single point
            if distance > shake_detection_threshold:
                cv2.circle(painter, centroid, size_brush, color, -1)
            else:
                cv2.line(painter, last_point, centroid, color, size_brush, -1)
        last_point = centroid


        if args["video_canvas"]:
            mask = np.not_equal(cv2.cvtColor(painter, cv2.COLOR_BGR2GRAY), 255)
            # Repeat mask along the three channels
            mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
            output = image.copy()
            output[mask] = painter[mask]
        else:
            output = painter

        # flip camera and drawing horizontaly
        cam_output = cv2.flip(cam_output, 1)  
        output = cv2.flip(output, 1)  

        cv2.imshow(window_name, cam_output)
        cv2.imshow(window_name_paint, output)
        cv2.imshow(window_name_segmented, image_p)
        cv2.imshow(window_name_area, mr.astype(np.uint8)*255)

        # user quits
        if k == ord("q"):
            break

        # brush
        elif k == ord("+"):
            size_brush += 1
        elif k == ord("-"):
            size_brush = max(2, size_brush-1)

        # color
        elif k == ord("r"):
            color = (0,0,255)
        elif k == ord("g"):
            color = (0,255,0)
        elif k == ord("b"):
            color = (255,0,0)

        # clear
        elif k == ord("c"):
            painter = np.ones((frame.shape[0], frame.shape[1], 3), np.uint8) * 255
        
        # save
        elif k == ord("w"):
            cv2.imwrite(f"drawing_{(time.ctime(time.time())).replace(' ', '_')}.png", painter)


    # end capture and destroy windows
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()