import cv2, time, math, winsound
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

EAR_THRESHOLD = 0.25
DROWSY_TIME = 2.5

MODEL_PATH = "models/face_landmarker.task"

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

face_landmarker = FaceLandmarker.create_from_options(options)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def calculate_ear(eye):
    A = math.dist(eye[1], eye[5])
    B = math.dist(eye[2], eye[4])
    C = math.dist(eye[0], eye[3])
    return (A + B) / (2.0 * C)

cap = cv2.VideoCapture(0)
timestamp = 0
eye_closed_start = None
alarm_on = False

def generate_frames():
    global timestamp, eye_closed_start, alarm_on

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = face_landmarker.detect_for_video(mp_image, timestamp)
        timestamp += 33

        status = "AWAKE"

        if result.face_landmarks:
            h, w, _ = frame.shape
            landmarks = result.face_landmarks[0]

            left_eye, right_eye = [], []

            for i in LEFT_EYE:
                lm = landmarks[i]
                left_eye.append((int(lm.x * w), int(lm.y * h)))
            for i in RIGHT_EYE:
                lm = landmarks[i]
                right_eye.append((int(lm.x * w), int(lm.y * h)))

            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0

            if ear < EAR_THRESHOLD:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                elif time.time() - eye_closed_start >= DROWSY_TIME:
                    status = "DROWSY"
                    if not alarm_on:
                        winsound.Beep(2500, 1000)
                        alarm_on = True
            else:
                eye_closed_start = None
                alarm_on = False

            cv2.putText(frame, f"EAR: {ear:.2f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if status == "DROWSY":
            cv2.rectangle(frame, (10, 70), (520, 130), (0, 0, 255), -1)
            cv2.putText(frame, "DROWSY! PLEASE WAKE UP",
                        (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        else:
            cv2.rectangle(frame, (10, 70), (350, 130), (0, 255, 0), -1)
            cv2.putText(frame, "STATUS: AWAKE",
                        (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
