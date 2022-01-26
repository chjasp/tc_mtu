import numpy as np
from djitellopy import tello
import cv2
from time import sleep

me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()


cap = cv2.VideoCapture(1)

# Adjust those values using the ColorPicker before every start of the script
hsvVals = [0,0,71,179,255,255]

# Global variables
threshold = 0.2  # percent of pixels needed to classify part of the image as having a line in it
senstivity = 2  # sensitive for left/right corrections (higher number makes the drone less sensitive)
weights = [-17, -10, 0, 10, 17]
fSpeed = 8
curve = 0


sensors = 3
width, height = 480, 360


# classify pixels as line / not line
def thresholding(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([hsvVals[0], hsvVals[1], hsvVals[2]])
    upper = np.array([hsvVals[3], hsvVals[4], hsvVals[5]])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_not(mask)
    return mask


# compute and show contours of line
def getContours(imgThres, img):
    cx = 0
    contours, hieracrhy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)
        cx = x + w // 2
        cy = y + h // 2
        cv2.drawContours(img, biggest, -1, (255, 0, 255), 7)
        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
    return cx


def getSensorOutput(imgThres, sensors):
    imgs = np.hsplit(imgThres, sensors)
    totalPixels = (img.shape[1] // sensors) * img.shape[0]
    senOut = []
    for x, im in enumerate(imgs):
        pixelCount = cv2.countNonZero(im)
        if pixelCount > threshold * totalPixels:
            senOut.append(1)
        else:
            senOut.append(0)
        cv2.imshow(str(x), im) # debugging of sensor output
    print(senOut)
    return senOut


def sendCommands(senOut, cx, forward):
    global curve
    # Translation / roll
    lr = (cx - height // 2) // senstivity  # If we want to fly sideways
    lr = int(np.clip(lr, -9, 9))
    if lr < 2 and lr > -2: lr = 0

    # Rotation / yaw
    if senOut == [1, 0, 0]: curve = weights[0]
    elif senOut == [1, 1, 0]: curve = weights[1]
    elif senOut == [0, 1, 0]: curve = weights[2]
    elif senOut == [0, 1, 1]: curve = weights[3]
    elif senOut == [0, 0, 1]: curve = weights[4]

    elif senOut == [0, 0, 0]: curve = weights[2]
    elif senOut == [1, 1, 1]: curve = weights[2]
    elif senOut == [1, 0, 1]: curve = weights[2]

    # adjusts speed and direction yaw (curve) based on drones travel direction
    if forward:
        fSpeedCur = -fSpeed
        curveCur = curve
    else:
        fSpeedCur = fSpeed
        curveCur = -curve

    me.send_rc_control(fSpeedCur, lr, 0, curve)
    print("Commands: -Right|+Left: ", fSpeed, ", -Backwards|+Forwards: ", lr, ", -Rotate_Left|+Rotate_Right: ", curveCur)



qrDecoder = cv2.QRCodeDetector()

hasnottakenoff = True
forward = True
while True:
    img_org = me.get_frame_read().frame
    img_2 = cv2.resize(img_org, (width, height))
    img_2 = cv2.flip(img_2, 0)
    img = cv2.rotate(img_2, cv2.cv2.ROTATE_90_CLOCKWISE) # rotate to fly sideways
    imgThres = thresholding(img)

    cx = getContours(imgThres, img)  # For Translation
    senOut = getSensorOutput(imgThres, sensors)  # Rotation

    sendCommands(senOut, cx, forward)
    cv2.imshow("Output", img)
    cv2.imshow("Path", imgThres)

    # Detect and decode the qrcode
    data, points, _ = qrDecoder.detectAndDecode(img_2)
    if points is not None and data == "Fly high" and forward:
        me.send_rc_control(0, 0, 0, 0)
        sleep(0.5)
        me.send_rc_control(0, 0, 22, 0)
        sleep(2.5)
        me.send_rc_control(0, 0, 0, 0)
        forward = False

    cv2.waitKey(1)

    # Takeoff sequence to increase the altitude upon takeoff
    if hasnottakenoff:
        hasnottakenoff = False
        sleep(6)
        me.takeoff()
        sleep(2)
        me.send_rc_control(0, 0, 25, 0)
        sleep(2.2)
        me.send_rc_control(0, 0, 0, 0)

tello.send_rc_control(0, 0, 0, 0)
tello.land()