import sys
import argparse

import cv2
print(cv2.__version__)

def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*10))    # added this line 
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        cv2.imwrite(pathOut + "/frame%d.jpg" % count, image)     # save frame as JPEG file
        count = count + 1


pathIn = '/Users/jaycheng/workspace/obj_detect_text/avi01.avi'
pathOut = '/Users/jaycheng/workspace/obj_detect_text/frame/avi'

# pathIn = 'videoplayback.mp4'
# pathOut = '/Users/jaycheng/workspace/obj_detect_text/frame/mp4'
extractImages(pathIn, pathOut)


# if __name__=="__main__":
#     a = argparse.ArgumentParser()
#     a.add_argument("--pathIn", help="path to video")
#     a.add_argument("--pathOut", help="path to images")
#     args = a.parse_args()
#     print(args)
#     extractImages(args.pathIn, args.pathOut)