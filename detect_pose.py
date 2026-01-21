import cv2
import math
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


def get_angle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    ang = ang + 360 if ang < 0 else ang
    return 360 - ang if ang > 180 else ang


CAMERA_WIDTH = 720
CAMERA_HEIGHT = 1280
WRITER_WIDTH = 640
WRITER_HEIGHT = 1136


class GesturedVideoCapture:
    is_recording: bool = False
#    video_writer: cv2.VideoWriter | None = None
    video_writer: cv2.VideoWriter
    capture_gesture_start_time: float = 0.0

    GESTURE_HOLD_TIME = 2.0 # 2 seconds

    def process_frame(self, frame, keypoints, current_time):
        nose = keypoints[0]
        nose_seen = nose[0] > 0 and nose[1] > 0
        left_ear_seen = keypoints[3][0] > 0 and keypoints[3][1] > 0
        right_ear_seen = keypoints[4][0] > 0 and keypoints[4][1] > 0

        left_wrist = keypoints[9]
        right_wrist = keypoints[10]

        in_capture_gesture = (
            nose_seen and left_ear_seen and right_ear_seen and
            right_wrist[1] < nose[1] < left_wrist[1]
        )

        if self.is_recording and self.video_writer is not None:
            resized = cv2.resize(frame, (WRITER_WIDTH, WRITER_HEIGHT))
            self.video_writer.write(resized)

        if self.capture_gesture_start_time:
            if in_capture_gesture:
                if self.capture_gesture_start_time + self.GESTURE_HOLD_TIME < current_time:
                    # Hold for GESTURE_HOLD_TIME seconds - starting or stopping capture
                    if self.is_recording:
                        self.stop_and_save_capture()
                    else:
                        self.start_capture()

                    return
            else:
                self.capture_gesture_start_time = 0
        elif in_capture_gesture:
                self.capture_gesture_start_time = current_time


    def start_capture(self):
        print("Start capturing")
        self.is_recording = True
        self.capture_gesture_start_time = 0
        self.video_writer = cv2.VideoWriter(
            f"output_{int(time.time())}.mp4",
            cv2.VideoWriter_fourcc(*'avc1'),
            10,
            (WRITER_WIDTH, WRITER_HEIGHT)
        )


    def stop_and_save_capture(self):
        print("Stop capturing")
        self.is_recording = False
        self.capture_gesture_start_time = 0
        self.video_writer.release()
        self.video_writer = None


def start_capture():
    cap = cv2.VideoCapture(0)
    model = YOLO('yolo11n-pose_full_integer_quant_edgetpu.tflite', task='pose')

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    last_frame_time_seconds = int(time.time())
    frames_per_last_second = 0
    fps = 0

    results = None
    # skip_detection_next_frame = False
    video_capture = GesturedVideoCapture()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            continue

        #results = model(frame)
        results = model.predict(frame, imgsz=256, conf=0.2, iou=0.45, half=True, verbose=False, device='tpu')

        # if skip_detection_next_frame:
        #     skip_detection_next_frame = False
        # else:
        #     results = model(frame)
        #     skip_detection_next_frame = True

        if not results:
            continue

        result = results[0]
        frames_per_last_second += 1

        keypoints = result.keypoints.xy.tolist()
        if not keypoints:
            continue

        keypoints = keypoints[0]
        if not keypoints:
            continue

        annotator = Annotator(frame)
        annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)

        annotated_frame = annotator.result()

        draw_angles(annotated_frame, keypoints)
        video_capture.process_frame(annotated_frame, keypoints, time.time())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_time = int(time.time())
        if frame_time > last_frame_time_seconds:
            last_frame_time_seconds = frame_time
            print("FPS:", frames_per_last_second)
            fps = frames_per_last_second
            frames_per_last_second = 0

        cv2.putText(
            annotated_frame,
            f"FPS: {fps}",
            (10, 20),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (25, 255, 25),
            1
        )
        cv2.imshow("YOLO Inference", annotated_frame)

    cap.release()
    cv2.destroyAllWindows()


def draw_angles(annotated_frame, keypoints):
    nose_seen = keypoints[0][0] > 0 and keypoints[0][1] > 0
    left_ear_seen = keypoints[3][0] > 0 and keypoints[3][1] > 0
    right_ear_seen = keypoints[4][0] > 0 and keypoints[4][1] > 0

    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]

    if left_ear_seen and not right_ear_seen:
        angle_knee = get_angle(left_hip, left_knee, left_ankle)
        angle_hip = get_angle(left_shoulder, left_hip, left_knee)
    else:
        angle_knee = get_angle(right_hip, right_knee, right_ankle)
        angle_hip = get_angle(right_shoulder, right_hip, right_knee)

    knee_label_coordinates = [int(c) for c in left_knee]
    knee_label_coordinates[0] += 10
    knee_label_coordinates[1] += 10
    cv2.putText(
        annotated_frame,
        f"{int(angle_knee)}",
        knee_label_coordinates,
        cv2.FONT_HERSHEY_PLAIN,
        1.5,
        (25, 25, 255),
        2
    )

    hip_label_coordinates = [int(c) for c in left_hip]
    hip_label_coordinates[0] += 10
    hip_label_coordinates[1] += 10
    cv2.putText(
        annotated_frame,
        f"{int(angle_hip)}",
        hip_label_coordinates,
        cv2.FONT_HERSHEY_PLAIN,
        1.5,
        (25, 25, 255),
        2
    )


def main():
    start_capture()


if __name__ == '__main__':
    main()
