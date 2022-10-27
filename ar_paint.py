#!/usr/bin/env python

# ---------- Augmented Reality Paint ------------
# Filipe Goncalves, 98083
# Diogo Monteiro, 97606
# Fábio Silva, 97729
# 
# PSR, October 2022.
# -----------------------------------------------

import cv2


def main():

    # start video capture
    capture = cv2.VideoCapture(0)

    # windows
    window_name = 'Original'
    cv2.namedWindow(window_name,cv2.WINDOW_AUTOSIZE)

    # while user wants video capture
    while True:

        # get frame
        ret, image = capture.read()

        # get key
        k = cv2.waitKey(1)

        # error getting the frame
        if not ret:
            print("failed to grab frame")
            break

        cv2.imshow(window_name, image)

        # user quits
        if k == ord("q"):
            break

    # end capture and destroy windows
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()