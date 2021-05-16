#!/usr/local/bin/python3
import cv2

videoCapture = cv2.VideoCapture('test.avi')

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.mp4', fourcc, 5.0, (1280, 720))

while(1):
    ret, frame=videoCapture.read()
    if frame is None:
        break
    
out.write(frame) # Write out frame to video
cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()