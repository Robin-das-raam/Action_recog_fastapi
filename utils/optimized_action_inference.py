import cv2
import os
import pandas as pd
from ultralytics import YOLO
import xgboost as xgb
import numpy as np
import cvzone

# TEMP_DIR = "./action_recog_backend/temp_output"
# os.makedirs(TEMP_DIR, exist_ok=True)

def detect_action(input_path, output_path,skip_rate=3):
    model_yolo = YOLO('yolo11s-pose.pt')
    model = xgb.Booster()
    model.load_model('/home/robinpc/Desktop/FastApi_prac/action_recog_backend/models_file/new_tuned_trained_model.json')

    # temp_output_path = os.path.join(
    #     TEMP_DIR, os.path.basename(output_path) + ".part"
    # )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    temp_path = output_path + ".temp.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'h264')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # width = 640
    # height = 720
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Width",width,"Height",height)
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    # print(f"✅ Writing to temporary file: {temp_output_path}")

    frame_count = 0
    last_annotations = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print(f"Reached end of video or failed to read frame at count {frame_count}")
            break

        frame_count += 1
        annotated_frame = frame.copy()

        if frame_count % skip_rate == 0:
            print(f"Processing frame {frame_count}")
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

    # processing_status[filename] = False

    print("complete")


    import subprocess
    subprocess.run([
        'ffmpeg',
        '-y', # Overwrite without asking
        '-i', temp_path,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        '-crf', '23',  # Quality balance (18-28, lower=better)
        output_path
    ])

    os.remove(temp_path) # Delete temporary file