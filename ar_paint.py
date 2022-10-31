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
import random
from colorama import Fore, Back, Style

# processes the image to give us the binary image to calculate the centroid of the largest area
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

# create numeric paint and the evaluation value
def numericPainter(painter, w, h, last_color):
    # create paint numeric
    evaluation_painter = painter.copy()

    # random number of lines
    num_lines = random.randint(2,2)

    # array for all the points necessary to build the lines and fill the gaps
    points = []

    # colors used (b,g,r)
    colors = [(255,0,0), (0,255,0), (0,0,255)]

    # calculate lines
    for i in range(num_lines+1):

        # all lines and last line which is the final wall
        if i != num_lines:
            points.append([(random.randint(int(w/num_lines)*i,int(w/num_lines)*(i+1)), 0), (random.randint(int(w/num_lines)*i,int(w/num_lines)*(i+1)), w)])
        else:
            points.append([[w,0], [w,h]])

        # start and end point for the polygon
        if i == 0:
            start_point = [0,0]
            end_point = [0,h]
        else:
            start_point = points[i-1][0]
            end_point = points[i-1][1]

        # all points for the polygon
        pts = np.array([start_point, points[i][0], points[i][1], end_point], np.int32)
        pts = pts.reshape((-1, 1, 2))

        # color randomizer for each gap
        color = colors[random.randint(0,2)]
        while last_color == color:
            color = colors[random.randint(0,2)]

        # create polygon to paint the right colors
        evaluation_painter = cv2.fillPoly(evaluation_painter, [pts], color)
        # points for the text in the center
        point_txt = (int((points[i][1][0] - start_point[0])/2))+start_point[0], int((end_point[1] - start_point[1])/2)

        # painter
        painter = cv2.putText(painter, str(colors.index(color)+1), point_txt, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

        # save last color for next polygon
        last_color = color

    # create lines
    for i in range(len(points)):
        cv2.line(evaluation_painter, points[i][0], points[i][1], (0,0,0), 3, -1)
        cv2.line(painter, points[i][0], points[i][1], (0,0,0), 3, -1)

    # temp for clear
    temp = painter.copy()

    # number of pixels for the evaluation
    total_pixels = np.sum(np.equal(evaluation_painter, painter).astype(np.uint8))

    # print different colors and their number
    print("Colors:\n"+ Style.BRIGHT + Fore.BLUE +"1 - Blue\n"+ Style.BRIGHT + Fore.GREEN +"2 - Green\n"+ Style.BRIGHT + Fore.RED +"3 - Red" + Style.RESET_ALL)

    # return variables
    return painter, evaluation_painter, temp, total_pixels

# create image for commands
def commands(canvas):
    # text variables
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2
    
    commands = ["q -> quit", "+ -> increase brush size", "- -> decrease brush size", "r -> change color to red",
                "g -> change color to green", "b -> change color to blue", "c -> clear canvas", 
                "w -> write image in file", "e -> erase", "s -> switch brush on/off"]

    for c in range(len(commands)):
        org = (50, 40*(c+1))
        canvas = cv2.putText(canvas, commands[c], org, font, fontScale, color, thickness, cv2.LINE_AA)

    return canvas

def main():

    # parse arguments
    parser = argparse.ArgumentParser(description='Definition of test mode')
    parser.add_argument('-j', '--json', required=True, dest='JSON', help='Full path to json file.')
    parser.add_argument('-v', '--video-canvas', required=False, help='Use video streaming as canvas', action="store_true", default=False)
    parser.add_argument('-p', '--paint-numeric', required=False, help='Use a numerical canvas to paint', action="store_true", default=False)
    parser.add_argument('-s', '--shake-detection', required=False, help='Use shake detection', action="store_true", default=False)

    args = vars(parser.parse_args())
    print(args)

    # Print the game (Typing Test) and the group membres
    print('\n---------------------------------')
    print('| ' + Style.BRIGHT + Fore.RED + 'PSR ' + Fore.GREEN + 'Augmented Reality Painter' + ' |' + Style.RESET_ALL)
    print('|                               |')
    print('|        Filipe Goncalves       |')
    print('|         Diogo Monteiro        |')
    print('|           Fábio Silva         |')
    print('|                               |')
    print('---------------------------------\n')

    # print counter
    count = 0

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
    window_name_commands = 'Commands'
    cv2.namedWindow(window_name_commands,cv2.WINDOW_AUTOSIZE)

    # Opening JSON file
    f = open(args["JSON"])
    data = json.load(f)

    # painter variables
    _, frame = capture.read()
    size_brush = 5
    last_point = None
    brush = True

    # height and width of our captured frame
    h, w = frame.shape[0:2]
    # starter color
    color = (0,0,0)

    # create painter
    painter = np.ones((h, w, 3)) * 255

    if args["paint_numeric"]:
        # create paint numeric
        painter, evaluation_painter, temp, total_pixels = numericPainter(painter, w, h, color)



    mr = np.ones((frame.shape[0], frame.shape[1], 3)) * 0



    # while user wants video capture
    while True:

        # get frame
        ret, image = capture.read()

        # get key
        k = cv2.waitKey(1)

        # error getting the frame
        if not ret:
            print(Style.BRIGHT + Fore.RED + "failed to grab frame" + Style.RESET_ALL)
            break

        # process image
        image_p, mask = processImage(data["limits"], image)

        connectivity = 4  
        # Perform the operation
        nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(cv2.cvtColor(image_p, cv2.COLOR_BGR2GRAY), connectivity, cv2.CV_32S)
        # Find the largest non background component.
        # Note: range() starts from 1 since 0 is the background label.
        if nb_components > 1:
            count = 0
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
            if last_point != None and brush:
                cv2.line(painter, last_point, (int(centroids[max_label][0]), int(centroids[max_label][1])), color, size_brush, -1)
            last_point = (int(centroids[max_label][0]), int(centroids[max_label][1]))
        else:
            if count == 0:
                print(Style.BRIGHT + Fore.RED + "Please place your object in front of the camera!" + Style.RESET_ALL)
                count += 1

        if args["video_canvas"]:
            mask = np.not_equal(cv2.cvtColor(painter, cv2.COLOR_BGR2GRAY), 255)
            image[mask] = painter
            painter = image

        # show images
        cv2.imshow(window_name, image)
        cv2.imshow(window_name_paint, painter)
        cv2.imshow(window_name_commands, commands(np.ones((frame.shape[0], frame.shape[1], 3)) * 0))
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
            painter = np.ones((frame.shape[0], frame.shape[1], 3)) * 255
            if args["paint_numeric"]:
                painter = temp
        
        # save
        if k == ord("w"):
            cv2.imwrite(f"drawing_{(time.ctime(time.time())).replace(' ', '_')}.png", painter)
            if args["paint_numeric"]:
                # do evaluation_painter
                max_pixels = (frame.shape[0] * frame.shape[1] * 3) - total_pixels
                total_pixels = np.sum(np.equal(evaluation_painter, painter).astype(np.uint8)) - total_pixels

                accuracy = ((total_pixels / max_pixels) * 100)

                print(f"Accuracy: {round(accuracy,2)}%")

                if accuracy < 40:
                    print("Evaluation: "+ Style.BRIGHT + Fore.RED +"Not Sattisfactory - D" + Style.RESET_ALL)
                elif accuracy < 60:
                    print("Evaluation: " + Style.BRIGHT + Fore.CYAN +"Satisfactory - C" + Style.RESET_ALL)
                elif accuracy < 80:
                    print("Evaluation: " + Style.BRIGHT + Fore.BLUE +"Good - B" + Style.RESET_ALL)
                elif accuracy < 90:
                    print("Evaluation: " + Style.BRIGHT + Fore.GREEN +"Very Good - A" + Style.RESET_ALL)
                else:
                    print("Evaluation: " + Style.BRIGHT + Fore.YELLOW +"Excellent - A+" + Style.RESET_ALL)

                cv2.destroyAllWindows()
                cv2.imshow("Evaluation", evaluation_painter)
                cv2.imshow(window_name_paint, painter)
                cv2.waitKey(0)
                break

        # extra keys
        # erase
        if k == ord("e"):
            color = (255,255,255)

        # switch brush
        if k == ord("s"):
            brush = False if brush else True


    # end capture and destroy windows
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()