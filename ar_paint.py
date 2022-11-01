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
from colorama import Fore, Style
import random

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
    parser.add_argument('-s', '--use-shake-detection', required=False, help='Use shake detection', action="store_true", default=False)
    parser.add_argument('-m', '--use-mouse', required=False, help='Use mouse as brush instead of centroid', action="store_true", default=False)

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
    height, width = frame.shape[0:2]
    # starter color
    color = (0,0,0)

    # create painter
    painter = np.ones((height, width, 3), np.uint8) * 255

    # min distance_squared between two consecutive points to be detected as an oscilation
    shake_detection_threshold = 1600

    # mask
    mr = np.zeros((height, width, 3))

    # mouse coordinates for mouse brush
    mouse_coords = {'x': None, 'y': None}

    if args['use_mouse']:
        # pass {'x': int, 'y': int} dict as param
        def mouseHoverCallback(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                # mirror x coordinate, as the drawing is flipped horizontally
                param['x'] = width - int(x)
                param['y'] = int(y)
        cv2.setMouseCallback(window_name_paint, mouseHoverCallback, mouse_coords)

    if args["paint_numeric"]:
        # create paint numeric
        painter, evaluation_painter, temp, total_pixels = numericPainter(painter, width, height, color)


    # while user wants video capture
    while True:

        # get frame
        ret, image = capture.read()
        cam_output = image

        # get key
        k = cv2.waitKey(1)

        # error getting the frame
        if not ret:
            print(Style.BRIGHT + Fore.RED + "failed to grab frame" + Style.RESET_ALL)
            break

        image_p, mask = processImage(data["limits"], image)

        # centroid initialization
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
            count = 0
            max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])

            # centroid coordinates
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
            if count == 0:
                print(Style.BRIGHT + Fore.RED + "Please place your object in front of the camera!" + Style.RESET_ALL)
                count += 1

        if last_point is not None and centroid is not None:
            # get squared distance between current point and previous
            distance = (last_point[0]-centroid[0])**2 + (last_point[1]-centroid[1])**2

            # oscilation detected, draw a single point
            if distance > shake_detection_threshold and args['use_shake_detection']:
                cv2.circle(painter, centroid, size_brush, color, -1)
            else:
                if brush:
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
        cv2.imshow(window_name_commands, commands(np.ones((frame.shape[0], frame.shape[1], 3)) * 0))
        cv2.imshow(window_name_segmented, image_p)
        cv2.imshow(window_name_area, mr.astype(np.uint8)*255)

        # user quits
        if k == ord("q"):
            print("Key Selected: "+Style.BRIGHT+Fore.YELLOW+"q"+Fore.RED+"\n\tEnding program")
            break

        # brush
        elif k == ord("+"):
            print("Key Selected: "+Style.BRIGHT+Fore.YELLOW+"+"+"\n\tIncreasing"+Style.RESET_ALL+" brush size")
            size_brush += 1
        elif k == ord("-"):
            print("Key Selected: "+Style.BRIGHT+Fore.YELLOW+"-"+"\n\tDecreasing"+Style.RESET_ALL+" brush size")
            size_brush = max(2, size_brush-1)

        # color
        elif k == ord("r"):
            print("Key Selected: "+Style.BRIGHT+Fore.YELLOW+"r"+Style.RESET_ALL+"\n\tChanging color to "+Style.BRIGHT+Fore.RED+"RED"+Style.RESET_ALL)
            color = (0,0,255)
        elif k == ord("g"):
            print("Key Selected: "+Style.BRIGHT+Fore.YELLOW+"g"+Style.RESET_ALL+"\n\tChanging color to "+Style.BRIGHT+Fore.GREEN+"GREEN"+Style.RESET_ALL)
            color = (0,255,0)
        elif k == ord("b"):
            print("Key Selected: "+Style.BRIGHT+Fore.YELLOW+"b"+Style.RESET_ALL+"\n\tChanging color to "+Style.BRIGHT+Fore.BLUE+"BLUE"+Style.RESET_ALL)
            color = (255,0,0)

        # clear
        elif k == ord("c"):
            print("Key Selected: "+Style.BRIGHT+Fore.YELLOW+"c"+Style.RESET_ALL+"\n\tClearing canvas")
            painter = np.ones((height, width, 3), np.uint8) * 255
            if args["paint_numeric"]:
                painter = temp
        
        # save
        elif k == ord("w"):
            file_name = f"drawing_{(time.ctime(time.time())).replace(' ', '_')}.png"
            print("Key Selected: "+Style.BRIGHT+Fore.YELLOW+"w"+Style.RESET_ALL+"\n\tWriting to file " + Style.BRIGHT + Fore.GREEN + file_name + Style.RESET_ALL)
            cv2.imwrite(file_name, output)
            if args["paint_numeric"]:
                # do evaluation_painter
                max_pixels = (frame.shape[0] * frame.shape[1] * 3) - total_pixels
                total_pixels = np.sum(np.equal(evaluation_painter, painter).astype(np.uint8)) - total_pixels

                accuracy = ((total_pixels / max_pixels) * 100)

                print("Accuracy: "+Style.BRIGHT+Fore.GREEN+round(accuracy,2)+Style.RESET_ALL+"%")

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
        elif k == ord("e"):
            print("Key Selected: "+Style.BRIGHT+Fore.YELLOW+"e"+Style.RESET_ALL+"\n\tErasing")
            color = (255,255,255)

        # switch brush
        elif k == ord("s"):
            brush = False if brush else True
            print("Key Selected: "+Style.BRIGHT+Fore.YELLOW+"s"+Style.RESET_ALL+"\n\tSwitching brush "+((Style.BRIGHT+Fore.GREEN+"ON") if brush else (Style.BRIGHT+Fore.RED+"OFF")) +Style.RESET_ALL)


    # end capture and destroy windows
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()