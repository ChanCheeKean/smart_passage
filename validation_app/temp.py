import cv2
cap = cv2.VideoCapture('rtsp://live:ThalesPass123@192.168.1.101:554/live', cv2.CAP_FFMPEG)
cnt = 0
cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)

while True:
    cnt, frame = cap.read()
    cv2.imshow("yolox", frame)
    ch = cv2.waitKey(1)
    if ch == 27 or ch == ord("q") or ch == ord("Q"):
        break
