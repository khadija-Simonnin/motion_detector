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

motion_frames = 0

while True:

    ret, frame2 = camera.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    frame1_gray = cv2.GaussianBlur(frame1_gray, (5, 5), 0)

    diff = cv2.absdiff(frame1_gray, gray)

    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    movement = cv2.countNonZero(thresh)
    
    if movement > MOTION_THRESHOLD:
        motion_frames += 1
    else:
        motion_frames = 0

    if motion_frames > 3:
        print("Real motion detected")

    if movement > MOTION_THRESHOLD:
        print("Motion detected!")
        cv2.putText(frame2, "MOTION DETECTED", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
    
    cv2.imshow("Motion Detector", thresh)

    frame1_gray = gray

    if cv2.waitKey(1) & 0xFF == 27:
        break
    
camera.release()
cv2.destroyAllWindows()