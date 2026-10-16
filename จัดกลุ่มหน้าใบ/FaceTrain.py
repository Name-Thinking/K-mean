import cv2
import os

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
name = 'parinya'
i = 1
os.mkdir(name)
while True:
    ret, frame = cap.read()
    cv2.rectangle(frame, (250,120), (390,300), (0,0,255), 2)
    face = cv2.cvtColor(frame[120:300, 250:390, :], cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)
    cv2.imshow('face', face)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(f'{name}/{i}.jpg', face)
        i += 1