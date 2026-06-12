import cv2
import csv
import time

MOTION_THRESHOLD = 2000

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Unable to access camera")
    exit()

ret, frame1 = camera.read()
if not ret:
    print("Failed to read camera")
    exit()

frame_h, frame_w = frame1.shape[:2]

detection_width = 400
detection_height = 300

detection_x = (frame_w - detection_width) // 2
detection_y = (frame_h - detection_height) // 2

frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame1_gray = frame1_gray[
    detection_y:detection_y + detection_height,
    detection_x:detection_x + detection_width
]

csv_file = open("events.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "event"])

motion_frames = 0
motion_active = False
previous_motion_state = False

recording = False
video_writer = None

while True:

    ret, frame2 = camera.read()
    if not ret:
        break

    gray_full = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    gray = gray_full[
        detection_y:detection_y + detection_height,
        detection_x:detection_x + detection_width
    ]

    # Noise reduction
    blur1 = cv2.GaussianBlur(frame1_gray, (5, 5), 0)
    blur2 = cv2.GaussianBlur(gray, (5, 5), 0)

    diff = cv2.absdiff(blur1, blur2)
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    movement = cv2.countNonZero(thresh)

    if movement > MOTION_THRESHOLD:
        motion_frames += 1
    else:
        motion_frames = max(0, motion_frames - 1)

    current_motion_state = motion_frames > 3
    
    if current_motion_state and not previous_motion_state:
        print("Motion START")
        csv_writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            "motion_start"
        ])
        csv_file.flush()

    elif not current_motion_state and previous_motion_state:
        print("Motion STOP")
        csv_writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            "motion_stop"
        ])
        csv_file.flush()

    previous_motion_state = current_motion_state

    if current_motion_state and not recording:
        recording = True
        video_writer = cv2.VideoWriter(
            "motion_output.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            20,
            (frame2.shape[1], frame2.shape[0])
        )
        print("Recording started")

    elif not current_motion_state and recording:
        recording = False
        video_writer.release()
        video_writer = None
        print("Recording stopped")

    if recording:
        video_writer.write(frame2)

    cv2.rectangle(
        frame2,
        (detection_x, detection_y),
        (detection_x + detection_width, detection_y + detection_height),
        (0, 255, 0),
        2
    )

    if recording:
        cv2.putText(
            frame2,
            "RECORDING",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("Motion Detector", frame2)

    frame1_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == 27:
        break

if video_writer is not None:
    video_writer.release()

camera.release()
csv_file.close()
cv2.destroyAllWindows()