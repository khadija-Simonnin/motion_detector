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
recording = False
video_writer = None

motion_active = False

while True:

    ret, frame2 = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    blur1 = cv2.GaussianBlur(frame1_gray, (5, 5), 0)
    blur2 = cv2.GaussianBlur(gray, (5, 5), 0)

    # Frame difference
    diff = cv2.absdiff(blur1, blur2)

    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    movement = cv2.countNonZero(thresh)

    if movement > MOTION_THRESHOLD:
        motion_frames += 1
    else:
        motion_frames = max(0, motion_frames - 1)

    if motion_frames > 3 and not motion_active:
        print("Real motion detected")
        motion_active = True

    elif motion_frames <= 3:
        motion_active = False

    # Start recording
    if motion_frames > 3 and not recording:
        recording = True
        video_writer = cv2.VideoWriter(
            "motion_output.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            20,
            (frame2.shape[1], frame2.shape[0])
        )
        print("Recording started")

    # Stop recording
    elif motion_frames <= 3 and recording:
        recording = False
        video_writer.release()
        video_writer = None
        print("Recording stopped")

    # Save frames
    if recording:
        video_writer.write(frame2)

    # Display
    cv2.imshow("Motion Detector", frame2)

    frame1_gray = gray

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()