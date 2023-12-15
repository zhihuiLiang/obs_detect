import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'size:{width}×{height}')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('origin_vedio.mp4',
                      fourcc=fourcc,
                      fps=30,
                      frameSize=(width, height))


def do_nothing(x):
    pass


window_name = 'hsv_threshold'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

#h: [0, 10] and [156, 180]
#s: [43, 255]
#v: [46, 255]
cv2.createTrackbar('h1_max', window_name, 0, 180, do_nothing)
cv2.createTrackbar('h1_min', window_name, 0, 180, do_nothing)
cv2.createTrackbar('h2_max', window_name, 0, 180, do_nothing)
cv2.createTrackbar('h2_min', window_name, 0, 180, do_nothing)
cv2.createTrackbar('s_max', window_name, 0, 255, do_nothing)
cv2.createTrackbar('s_min', window_name, 0, 255, do_nothing)
cv2.createTrackbar('v_max', window_name, 0, 255, do_nothing)
cv2.createTrackbar('v_min', window_name, 0, 255, do_nothing)

while True:
    _, frame = cap.read()
    cv2.imshow("A video", frame)

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

    cv2.imshow('Binary', binary_img)

    ret = out.write(frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

out.release()
cap.release()
cv2.destroyAllWindows()
