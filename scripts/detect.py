from math import fabs
import cv2
import numpy as np
import json
import socket
import sys

ENABLE_BLUR = False
USE_VIDEO = False
SAVE_VIDEO = False

# H1_MAX = 20
# H2_MIN = 143
# S_MIN = 51
# S_MAX = 201
# V_MIN = 186
# V_MAX = 255

H1_MAX = 30
H2_MIN = 140
S_MIN = 70
S_MAX = 255
V_MIN = 140
V_MAX = 255

BLUR_SIZE = 5
ROI_SIZE = (20, 28)

MIN_BOX_CONTOUR = 200
MIN_NUN_CONTOUR = 20
MAX_NUM_CONTOUR = 2000

RATIO = 1.195 / 228

USE_SOCKET = False
PORT = 9000
SERVER_IP = '192.168.1.142'

if not USE_VIDEO:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture('origin_video.mp4')
if not cap.isOpened():
    print("Cannot open camera or video")
    exit()

conf = {}
with open('conf/conf.json') as f:
    conf_data = json.load(f)

shape = conf_data['shape']
cam_mat = np.transpose(conf_data['cam_mat'])
distor_coffe = np.array(conf_data['distortion'])
rotate_mat = np.float32(conf_data['rotate_mat'])

if USE_SOCKET:
    try:
        sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error as msg:
        print('Failed to create socket. Error code: ' + str(msg[0]) + ', Error message : ' + msg[1]) 
        sys.exit()
    sk.connect((SERVER_IP, PORT))
    print("Connect Server Successfully")


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'size:{width}×{height}')

if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('origin_video.mp4',
                          fourcc=fourcc,
                          fps=30,
                          frameSize=(width, height))

# num_classify_net = cv2.dnn.readNetFromONNX('mlp.onnx')

def sortedPoints(points):
    length = []
    for i, p in enumerate(points):
        p2 = points[(i + 1) % 4]
        dist = (p[0] - p2[0]) * (p[0] - p2[0]) + (p[1] - p2[1]) * (p[1] -
                                                                   p2[1])
        length.append([dist, i])
    length = sorted(length, key=lambda l: l[0])
    sorted_points = []
    shortest_index = length[0][1]
    for _ in range(4):
        sorted_points.append(points[shortest_index % 4])
        shortest_index += 1
    return sorted_points


def findConvexPoints(hull):
    x_min_index = np.argmin(hull[:, :, 0])
    x_max_index = np.argmax(hull[:, :, 0])
    y_min_index = np.argmin(hull[:, :, 1])
    y_max_index = np.argmax(hull[:, :, 1])
    index = [x_min_index, y_min_index, x_max_index, y_max_index]
    points = [hull[i][0] for i in index]

    return sortedPoints(points)


def drawConvex(points, img, color):
    assert (len(points) == 4)
    for i, p in enumerate(points):
        p1 = (int(p[0]), int(p[1]))
        p2 = (int(points[(i + 1) % 4][0]), int(points[(i + 1) % 4][1]))
        cv2.line(img, p1, p2, color, 1)
        cv2.putText(img, str(i), p1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                    1)


def extractNumImg(points, img):
    width, height = ROI_SIZE
    target_vertices = np.float32([[0, 0], [width, 0], [width, height],
                                  [0, height]])

    rotate_mat = cv2.getPerspectiveTransform(np.float32(points),
                                             target_vertices)
    perspective_img = cv2.warpPerspective(img, rotate_mat, ROI_SIZE)
    cv2.imshow('perspective_img', perspective_img)

    num_img = cv2.cvtColor(perspective_img, cv2.COLOR_RGB2GRAY)
    _, num_img = cv2.threshold(num_img, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return num_img


# def classifyNum(img):
#     img = np.float32(img) / 255

#     def classify(img):
#         blob = cv2.dnn.blobFromImage(img, 1.0, ROI_SIZE)
#         num_classify_net.setInput(blob)

#         outputs = num_classify_net.forward()

#         max_prob = np.max(outputs)
#         softmax_prob = cv2.exp(outputs - max_prob)
#         sum = cv2.sumElems(softmax_prob)[0]
#         softmax_prob = softmax_prob / sum

#         _, conf, _, cls_point = cv2.minMaxLoc(softmax_prob)
#         return cls_point[0], conf

#     id1, conf1 = classify(img)
#     id2 = 0
#     conf2 = 0.0
#     if id1 == 8 or conf1 < 0.7:
#         img = cv2.flip(img, 1)
#         id2, conf2 = classify(img)
#     return id1 if conf1 > conf2 or id2 == 8 else id2


def doPnP(idx, points_in_img):
    width, height = shape[idx - 1]
    points_in_world = np.float32([[-width / 2, +height / 2, 0.0],
                                  [+width / 2, +height / 2, 0.0],
                                  [+width / 2, -height / 2, 0.0],
                                  [-width / 2, -height / 2, 0.0]])
    return cv2.solvePnP(points_in_world, np.float32(points_in_img),
                        np.float32(cam_mat), np.float32(distor_coffe), None,
                        None, False, cv2.SOLVEPNP_EPNP)


if __name__ == '__main__':
    seq = 0
    while True:
        _, frame = cap.read()
        frame = cv2.undistort(frame, cam_mat, distor_coffe)
        height, width, _ = frame.shape
        result_img = np.copy(frame)

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask0 = cv2.inRange(hsv_img, np.array([0, S_MIN, V_MIN]),
                            np.array([H1_MAX, S_MAX, V_MAX]))
        mask1 = cv2.inRange(hsv_img, np.array([H2_MIN, S_MIN, V_MIN]),
                            np.array([255, S_MAX, V_MAX]))

        binary_img = mask0 + mask1
        cv2.imshow('Binary', binary_img)
        blur_img = binary_img
        if ENABLE_BLUR:
            blur_img = cv2.blur(binary_img, (BLUR_SIZE, BLUR_SIZE))
            cv2.imshow('Blur', blur_img)

        obs_loc_arr = []
        binary_map = np.ones((height, width, 3), dtype=np.uint8) * 255
        contour_img = np.copy(frame)
        contours, hierachy = cv2.findContours(blur_img, cv2.RETR_CCOMP,
                                              cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)
        cv2.imshow('contour', contour_img)
        for i, contour in enumerate(contours):
            contour_area = cv2.contourArea(contour)
            if contour_area < MIN_NUN_CONTOUR or contour_area > MAX_NUM_CONTOUR or hierachy[
                    0][i][3] == -1:
                # print(f'box contourArea:{cv2.contourArea(contour)} too small or this contour do not have parenet contour')
                continue
            num_rect = cv2.minAreaRect(contour)
            num_hull = cv2.boxPoints(num_rect)
            num_points = sortedPoints(num_hull)
            drawConvex(num_points, result_img, (0, 255, 0))
            # num_image = extractNumImg(num_points, frame)
            # id = classifyNum(num_image)
            # cv2.putText(result_img, str(id),
            #             (int(num_points[0][0] + 10), int(num_points[0][1])),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

            box_contour_idx = hierachy[0][i][3]
            box_contour = contours[box_contour_idx]
            box_rect = cv2.minAreaRect(box_contour)
            box_hull = cv2.boxPoints(box_rect)
            box_points = sortedPoints(box_hull)
            drawConvex(box_points, result_img, (0, 255, 0))

            obs_loc_arr.append(box_points)

        cv2.imshow("origin", frame)
        cv2.imshow('result', result_img)
        obs_loc_msg = "SSeq-" + str(seq) + " "
        for obs_loc in obs_loc_arr:
            str_msg = f'obs:'
            for p in obs_loc:
                x = p[1] * RATIO
                y = p[0] * RATIO
                str_msg += f'({x:.3f},{y:.3f})'
            obs_loc_msg += " " +  str_msg
        obs_loc_msg += "ESeq-" + str(seq) + "\n"
        print(obs_loc_msg)
        if USE_SOCKET:
            sk.sendall(obs_loc_msg.encode())
        seq += 1

        if SAVE_VIDEO:
            ret = out.write(frame)

        c = cv2.waitKey(1)
        if c == 27:
            break
        
    if SAVE_VIDEO:
        out.release()
    cap.release()
    cv2.destroyAllWindows()