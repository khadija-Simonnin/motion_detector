import cv2

MOTION_THRESHOLD = 2000

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

    movement = cv2.countNonZero(thresh)

    if movement > MOTION_THRESHOLD:
        print("Motion detected!")
        cv2.putText(frame2, "MOTION DETECTED", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
    
    cv2.imshow("Motion Detector", frame2)

    frame1_gray = gray

    if cv2.waitKey(1) & 0xFF == 27:
        break
    
camera.release()
cv2.destroyAllWindows()