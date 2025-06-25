import cvzone
import cv2
import numpy as np
from cvzone.ColorModule import ColorFinder

#initialized the capture
cap = cv2.VideoCapture('Videos/vid (7).mp4')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)


#initialized the color finder
myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 0, 'smin': 137, 'vmin': 0, 'hmax': 30, 'smax': 255, 'vmax': 255}

#position list
posListX = []
posListY = []

#List of one of the coordinate which will go across a range of values covering
#the whole trajectory (here the range is the image pixels of the width of the input video), corresponding to this value (x axis)
#the position (y axis) of the ball will be calculated using the polynomial regression
xList = [item for item in range(int(width))]

#basket prediction
basket_prediction = False
start_prediction = False

while True:
    success, img = cap.read()
    # img = cv2.imread('Ball.png')
    img = img[0:900, :]

    #ball color detection
    imgColor, mask = myColorFinder.update(img, hsvVals)
    imgContour, contours = cvzone.findContours(img, mask, minArea=300)

    #ball position
    if contours:
        posListX.append(contours[0]['center'][0])
        # print(posListX)
        posListY.append(contours[0]['center'][1])


    if posListX and posListY:
        #polynomial regression fit (y=ax^2+bx+c quadratic for projectile motion)
        #get coefficients of the polynomial regression (i.e a,b,c)
        a,b,c = np.polyfit(posListX, posListY, 2)

        # drawing the position each frame from posList
        for i, (posX,posY) in enumerate(zip(posListX,posListY)):
            pos = (posX,posY)
            cv2.circle(imgContour, pos, 10, (255, 0, 255), cv2.FILLED)
            if i == 0:
                cv2.line(imgContour, pos, pos, (255, 0, 255), 5)
            else:
                cv2.line(imgContour, pos, (posListX[i - 1],posListY[i - 1]), (255, 0, 255), 5)

        # for x in xList:
        #     y_draw = int(a * x ** 2 + b * x + c)
        #     cv2.circle(imgContour, (x, y_draw), 2, (255, 0, 0), cv2.FILLED)

        #projectile path prediction and collision detection
        if len(posListX) >= 4:  # wait until we have at least 6 points

            # Fit parabola
            coeffs = a,b,c
            poly = np.poly1d(coeffs)

            # Predict y for all X
            y_pred_full = poly(posListX)

            # Calculate R²
            ss_res = np.sum((np.array(posListY) - y_pred_full) ** 2)
            ss_tot = np.sum((np.array(posListY) - np.mean(posListY)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            print(f"R²: {r_squared:.3f}")

            # Collision detection
            if r_squared < 0.997:
                stop_drawing = True
                print("Collision detected!")
                cv2.putText(imgContour, "Collision detected", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                # for x in xList:
                #     y_draw = int(a*x**2 + b*x + c)
                #     cv2.circle(imgContour, (x, y_draw), 2, (255, 0, 0), cv2.FILLED)
            else:
                stop_drawing = False
                print("No collision detected.")
                cv2.putText(imgContour, "collision not detected", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            # Draw predicted curve if no collision
            if not stop_drawing:
                for x in xList:
                    y_draw = int(a*x**2 + b*x + c)
                    cv2.circle(imgContour, (x, y_draw), 2, (0, 255, 0), cv2.FILLED)


    #basketball in hoop prediction:-

        #predict with only the first 10 points
        if (len(posListX))<10:
            # check at y = 590 pixel
            c = c - 590
            x = int((-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a))
            if 330 < x < 430:
                basket_prediction = True
        #check if the ball is in the basket pixel range (330 to 430)
        if basket_prediction:
            cvzone.putTextRect(imgContour, "in basket", (150, 150), thickness=2, offset=20)
            start_prediction = True
        elif not basket_prediction or start_prediction:
            cvzone.putTextRect(imgContour, "not in basket", (150, 150), thickness=2, offset=20)




        # #equation of the parabola used to get y
        # for x in xList:
        #     y = int(a*x**2 + b*x + c)
        #     cv2.circle(imgContour, (x,y), 2, (255, 0, 0), cv2.FILLED)


    #display
    # img = cv2.resize(img, (0, 0), None, 0.7, 0.7)
    # cv2.imshow("Video", img)
    # cv2.imshow("imgball", imgColor)
    imgContour = cv2.resize(imgContour, (0, 0), None, 0.7, 0.7)
    cv2.imshow("imgcontour", imgContour)


    #quit
    if cv2.waitKey(1) == ord("q"):
        cap.release()
        cv2.destroyAllWindows()

    #general waiting
    cv2.waitKey(100)