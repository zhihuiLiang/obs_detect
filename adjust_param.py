import cv2
import numpy as np
from detect import drawConvex
from detect import findConvexPoints

USE_VIDEO = True

if not USE_VIDEO:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture('origin_video.mp4')
if not cap.isOpened():
    print("Cannot open camera or video")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'size:{width}×{height}')


def doNothing(x):
    pass


brick_width, brick_height = 480, 480
four_corner_points = []
rotate_mat = np.float32()


def onMouseClick(event, x, y, flags, param):
    global rotate_mat
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        four_corner_points.append([x, y])
        print(f"Mouse clicked at (x={x}, y={y})")
    if len(four_corner_points) == 4:
        vertices = np.float32([[0, 0], [brick_width, 0],
                               [brick_width, brick_height], [0, brick_height]])

        rotate_mat = cv2.getPerspectiveTransform(
            np.float32(four_corner_points), vertices)
        print('rotate mat:', rotate_mat)


window_name = 'hsv_threshold'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

#h: [0, 10] and [156, 180]
#s: [43, 255]
#v: [46, 255]
cv2.createTrackbar('h1_min', window_name, 0, 180, doNothing)
cv2.createTrackbar('h1_max', window_name, 20, 180, doNothing)
cv2.createTrackbar('h2_min', window_name, 140, 180, doNothing)
cv2.createTrackbar('h2_max', window_name, 180, 180, doNothing)
cv2.createTrackbar('s_max', window_name, 255, 255, doNothing)
cv2.createTrackbar('s_min', window_name, 51, 255, doNothing)
cv2.createTrackbar('v_max', window_name, 255, 255, doNothing)
cv2.createTrackbar('v_min', window_name, 140, 255, doNothing)

_, frame = cap.read()

while True:
    cv2.imshow("frame", frame)
    cv2.setMouseCallback("frame", onMouseClick)

    if len(four_corner_points) == 4:
        perspective_img = cv2.warpPerspective(frame, rotate_mat,
                                              (brick_width, brick_height))
        cv2.imshow('perspective', perspective_img)

    contour_img = np.copy(frame)
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h1_max = cv2.getTrackbarPos('h1_max', window_name)
    h1_min = cv2.getTrackbarPos('h1_min', window_name)
    h2_max = cv2.getTrackbarPos('h2_max', window_name)
    h2_min = cv2.getTrackbarPos('h2_min', window_name)
    s_max = cv2.getTrackbarPos('s_max', window_name)
    s_min = cv2.getTrackbarPos('s_min', window_name)
    v_max = cv2.getTrackbarPos('v_max', window_name)
    v_min = cv2.getTrackbarPos('v_min', window_name)

    mask0 = cv2.inRange(hsv_img, np.array([h1_min, s_min, v_min]),
                        np.array([h1_max, s_max, v_max]))
    mask1 = cv2.inRange(hsv_img, np.array([h2_min, s_min, v_min]),
                        np.array([h2_max, s_max, v_max]))

    binary_img = mask0 + mask1

    contours, hierachy = cv2.findContours(binary_img, cv2.RETR_CCOMP,
                                          cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        hull = cv2.convexHull(contour)
        points = findConvexPoints(hull)
        drawConvex(points, contour_img)

    cv2.imshow('Binary', binary_img)
    cv2.imshow('contour', contour_img)

    c = cv2.waitKey(10)
    if c == 27:
        break

cv2.destroyAllWindows()
