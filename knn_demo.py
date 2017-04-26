# Pure numpy k-nearest neighbors demo with OpenCV visualization
# Written by Matt Zucker, April 2017

import numpy as np
import cv2

######################################################################
# Define some constants 

SIZE = 512
NUM_POINTS = 1000

######################################################################
# OpenCV has fast matching code, but the Python interface to it
# changes significantly from version to version. This is a reasonably
# fast pure numpy k-nearest-neighbor function that you might find
# helpful for your own code.

def bruteforce_knn(points, p, k):

    assert(len(p) == points.shape[1])

    diff = points - p
    d = (diff**2).sum(axis=1)
    idx = np.argpartition(d, k)

    idx = idx[:k]
    d = d[idx]

    idx2 = np.argsort(d)
    return idx[idx2], np.sqrt(d[idx2])

######################################################################
# Show a nice demo illustrating knn search

def main():

    # Sample a bunch more points than we actually need because we will do
    # rejection sampling below.
    points = np.random.random((NUM_POINTS*4, 2))*SIZE

    # Reject points not inside the "donut" we want to display
    diff = points - (SIZE/2, SIZE/2)
    d = np.sqrt((diff**2).sum(axis=1))
    donut_mask = (d < SIZE/2 - 8) & (d > SIZE/8)
    points = points[donut_mask]

    # Trim down to the desired number of points
    points = points[:NUM_POINTS].astype(np.float32)

    # Create a background image with each point
    background = 255*np.ones((SIZE, SIZE, 3), dtype='uint8')

    for p in points:
        cv2.circle(background, tuple(p.astype(int)), 3, (0, 0, 0), -1, 16)

    # Pop up a window
    cv2.namedWindow('knn')
    cv2.imshow('knn', background)

    ##################################################
    # Define a mouse handling function:
    
    def mouse(event, x, y, flags, param):

        # Create a point to match
        p = np.array([x,y], dtype=np.float32)

        # Do our brute-force knn search
        matches, dists = bruteforce_knn(points, p, 3)

        # Display the point and the neighbors
        display = background.copy()

        cv2.circle(display, (x, y), 3, (255, 0, 255), 1, 16)

        colors = [ (0, 0, 255),
                   (0, 255, 0),
                   (255, 0, 0) ]

        for color, i in zip(colors, matches):

            pi = tuple(points[i].astype(int))

            cv2.line(display, (x, y), pi, color, 1, 16)
            cv2.circle(display, pi, 4, color, -1, 16)

        cv2.imshow('knn', display)

    ##################################################
    # Install our mouse callback and run the demo
    
    cv2.setMouseCallback('knn', mouse, None)

    while True:    
        k = np.uint8(cv2.waitKey(5)).view(np.int8)
        if k == 27:
            break

if __name__ == '__main__':
    main()
