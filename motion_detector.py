import cv2
import csv
import time
import collections
import os

def beep_start():
    os.system("afplay /System/Library/Sounds/Ping.aiff")

def beep_stop():
    os.system("afplay /System/Library/Sounds/Bottle.aiff")

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
previous_motion_state = False

recording = False
video_writer = None

motion_count = 0
start_time = time.time()

history = collections.deque(maxlen=5)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:

    ret, frame2 = camera.read()
    if not ret:
        break

    gray_full = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    gray = gray_full[
        detection_y:detection_y + detection_height,
        detection_x:detection_x + detection_width
    ]

    blur1 = cv2.GaussianBlur(frame1_gray, (5, 5), 0)
    blur2 = cv2.GaussianBlur(gray, (5, 5), 0)

    diff = cv2.absdiff(blur1, blur2)
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    movement = cv2.countNonZero(thresh)

    motion_triggered = movement > MOTION_THRESHOLD

    history.append(motion_triggered)
    stable_motion = sum(history) > 3

    if stable_motion:
        motion_frames += 1
    else:
        motion_frames = max(0, motion_frames - 1)

    current_motion_state = motion_frames > 3

    if recording:
        status = "RECORDING"
    elif current_motion_state:
        status = "MOTION"
    else:
        status = "IDLE"

    current_time = time.time()

    if current_motion_state and not previous_motion_state:
        beep_start()
        csv_writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            "motion_start"
        ])
        csv_file.flush()
        motion_count += 1

    elif not current_motion_state and previous_motion_state:
        beep_stop()
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

    elif not current_motion_state and recording:
        recording = False
        video_writer.release()
        video_writer = None

    if recording:
        video_writer.write(frame2)

    cv2.rectangle(
        frame2,
        (detection_x, detection_y),
        (detection_x + detection_width, detection_y + detection_height),
        (0, 255, 0),
        2
    )

    cv2.rectangle(frame2, (10, 10), (260, 150), (0, 0, 0), -1)

    cv2.putText(frame2, f"STATUS: {status}", (20, 40),
                font, 0.7, (255, 255, 255), 2)

    cv2.putText(frame2, f"MOVES: {motion_count}", (20, 70),
                font, 0.7, (255, 255, 255), 2)

    cv2.putText(frame2, f"RECORDING: {recording}", (20, 100),
                font, 0.7, (255, 255, 255), 2)

    cv2.imshow("Motion Detector", frame2)

    frame1_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == 27:
        break

if video_writer is not None:
    video_writer.release()

camera.release()
csv_file.close()
cv2.destroyAllWindows()