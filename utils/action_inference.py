import cv2
import os
import pandas as pd
from ultralytics import YOLO
import xgboost as xgb
import numpy as np
import cvzone

def detect_action(input_path, output_path):
    model_yolo = YOLO('yolo11s-pose.pt')
    model = xgb.Booster()
    # model.load_model('./models_file/new_tuned_trained_model.json')
    model.load_model('/home/robinpc/Desktop/FastApi_prac/action_recog_backend/models_file/new_tuned_trained_model.json')

    cap =cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        results = model_yolo(frame, verbose = False)
        annotated_frame = frame.copy()

        for r in results:
            bound_boxes = r.boxes.xyxy
            conf = r.boxes.conf.tolist()
            keypoints = r.keypoints.xyn.tolist()

            for index, box in enumerate(bound_boxes):
                if conf[index] > 0.55:
                    x1,y1, x2, y2 = box.tolist()
                    data = {}
                    for j in range(len(keypoints[index])):
                        data[f'x{j}'] = keypoints[index][j][0]
                        data[f'y{j}'] = keypoints[index][j][1]

                    df = pd.DataFrame(data, index=[0])
                    # df.columns = df.columns.astype(str)
                    # df.columns = df.columns.str.replace(r'[\[\]<>]','', regex=True)
                    dmatrix = xgb.DMatrix(df)

                    pred_probs = model.predict(dmatrix)
                    class_idx = int(np.argmax(pred_probs[0]))
                    class_label = {0:"Giving_Action", 1:"Money_counting", 2:"Normal"}[class_idx]
                    confidence = float(np.max(pred_probs[0]))

                    color = (0, 255, 0) if class_label == "Normal" else (0, 0, 255)
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cvzone.putTextRect(
                        annotated_frame,
                        f"{class_label} ({confidence:.2f})",
                        (int(x1), int(y1)+ 50),
                        scale = 1,
                        thickness = 1
                    )

                
        out.write(annotated_frame)

    cap.release()
    out.release()
