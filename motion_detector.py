import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Unable to access camera")
    exit()

while True:

    success, frame = camera.read()

    if not success:
        break

    cv2.imshow("Motion Detector", frame)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()