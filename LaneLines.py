import cv2
import numpy as np

camera = cv2.VideoCapture(r'C:\Users\Kevin\Desktop\Lane\drive.mp4')
#camera = cv2.VideoCapture(0)

width = int(camera.get(3))
height = int(camera.get(4))

print("W:" + str(width) + " H:" + str(height))

# Region of Interest
pts1 = np.float32([[int((width/2) - 100), 1250], [int((width/2) + 100), 1250], [int((width/2) - 500), height], [int((width/2) + 500), height]])
pts2 = np.float32([[0, 0], [480, 0], [0, 720], [480, 720]])

def main():

    while camera.isOpened:

        ret, frame = camera.read()

        #Filtered Image
        filtered = filters(frame)

        #Warp Image
        warpped = warpImg(filtered)
        
        #Lane Lines
        lines = cv2.HoughLinesP(warpped, 2, np.pi/180, 100, np.array([]), minLineLength=100, maxLineGap=150)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        #Draw Dots in video
        for i in range(0, 4):
            cv2.circle(frame, (pts1[i][0], pts1[i][1]), 5, (0, 0, 255), cv2.FILLED)

        #Display
        cv2.imshow('Video', cv2.resize(frame, (1280, 720)))
        cv2.waitKey(1)

def filters(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = cv2.addWeighted(gray, 2, gray, 0, -200)

    maskWhite = cv2.inRange(contrast, 200, 255)
    maskYellow = cv2.inRange(contrast, 100, 255)
    combineMask = cv2.bitwise_and(maskWhite, maskYellow)

    blur = cv2.GaussianBlur(combineMask, (5, 5), 0)
    canny = cv2.Canny(blur, 75, 150)

    return canny

def warpImg(img):

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, matrix, (480, 720))

    #cv2.imshow('Video', warped)

    matrix = cv2.getPerspectiveTransform(pts2, pts1)
    warped = cv2.warpPerspective(warped, matrix, (width, height))

    return warped

main()