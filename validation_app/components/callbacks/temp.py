import cv2

video_loader = cv2.VideoCapture("rstp://service:Thales1$8o8@192.168.0.103/554/?h26x=0")
_, image = video_loader.read()
