import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Unable to access camera")
    exit()

ret, frame1 = camera.read()
if not ret:
    print("Failed to read camera")
    exit()

frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while True:

    ret, frame2 = camera.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(frame1_gray, gray)

    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    cv2.imshow("Motion Detector", thresh)

    frame1_gray = gray

    if cv2.waitKey(1) & 0xFF == 27:
        break
    
camera.release()
cv2.destroyAllWindows()