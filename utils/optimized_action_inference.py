import cv2
import os
import pandas as pd
from ultralytics import YOLO
import xgboost as xgb
import numpy as np
import cvzone

def detect_action(input_path, output_path, skip_rate=3):
    model_yolo = YOLO('yolo11s-pose.pt')
    model = xgb.Booster()
    model.load_model('/home/robinpc/Desktop/FastApi_prac/action_recog_backend/models_file/new_tuned_trained_model.json')

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    last_annotations = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        annotated_frame = frame.copy()

        if frame_count % skip_rate == 0:
            results = model_yolo(frame, verbose=False)
            last_annotations = []

            for r in results:
                bound_boxes = r.boxes.xyxy
                conf = r.boxes.conf.tolist()
                keypoints = r.keypoints.xyn.tolist()

                for index, box in enumerate(bound_boxes):
                    if conf[index] > 0.55:
                        x1, y1, x2, y2 = box.tolist()
                        data = {}
                        for j in range(len(keypoints[index])):
                            data[f'x{j}'] = keypoints[index][j][0]
                            data[f'y{j}'] = keypoints[index][j][1]

                        df = pd.DataFrame(data, index=[0])
                        dmatrix = xgb.DMatrix(df)

                        pred_probs = model.predict(dmatrix)
                        class_idx = int(np.argmax(pred_probs[0]))
                        class_label = {0: "Giving_Action", 1: "Money_counting", 2: "Normal"}[class_idx]
                        confidence = float(np.max(pred_probs[0]))

                        color = (0, 255, 0) if class_label == "Normal" else (0, 0, 255)
                        last_annotations.append((int(x1), int(y1), int(x2), int(y2), class_label, confidence, color))

        # reuse last_annotations
        for (x1, y1, x2, y2, class_label, confidence, color) in last_annotations:
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cvzone.putTextRect(
                annotated_frame,
                f"{class_label} ({confidence:.2f})",
                (x1, y1 + 50),
                scale=3,
                thickness=3
            )

        out.write(annotated_frame)

    cap.release()
    out.release()
    print("✅ Processing done. Output saved to:", output_path)

# Example usage
# detect_action('/path/to/input.mp4', '/path/to/output.mp4')
