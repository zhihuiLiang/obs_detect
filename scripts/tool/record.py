import cv2
import numpy as np
import os

SAVE_VIDEO = False

save_dir = 'record'
if not os.path.exists(save_dir):
    print(f'{save_dir} do not exist, make one')
    os.makedirs(save_dir)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'size:{width}×{height}')

if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_dir + 'origin_vedio.mp4',
                          fourcc=fourcc,
                          fps=30,
                          frameSize=(width, height))

cnt = 1
while True:
    _, frame = cap.read()
    cv2.imshow("A video", frame)

    if SAVE_VIDEO:
        ret = out.write(frame)
    c = cv2.waitKey(1)

    if c == 49:  #'1'
        print(f'Save picture {cnt}')
        pic_path = save_dir + '/' + str(cnt) + '.jpg'
        cv2.imwrite(pic_path, frame)
        cnt += 1
    if c == 27:
        break

if SAVE_VIDEO:
    out.release()
cap.release()
cv2.destroyAllWindows()