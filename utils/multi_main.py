import cv2
import os
import pandas as pd
from ultralytics import YOLO
import xgboost as xgb
import numpy as np
import cvzone

# Define the path to the video file
video_path = "/home/robinpc/Desktop/FastApi_prac/action_recog_backend/uploads/output_part_4.mp4"
# video_path = "/home/robin/Desktop/Doer_sec_sur_proj/videos/33_Md. ali Azam_Magura.  Jagla Bazar csp video futej.mkv"

# Label decoder (adjust based on training order)
label_decoder = {
    0: "Giving_Action",
    1: "Money_Counting",
    2: "Normal"
}

def detect_action(video_path):
    # Load YOLOv8 pose model
    model_yolo = YOLO('yolo11s-pose.pt')

    # Load trained XGBoost model
    model = xgb.Booster()
    model.load_model('/home/robinpc/Desktop/FastApi_prac/action_recog_backend/models_file/new_tuned_trained_model.json')
    # model.load_model('./multi_action_recognition/dataset/trained_model_1000.json')

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get video FPS
    delay = int(1000 / fps)               # Delay between frames in ms
    print(f"Total Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

    frame_tot = 0
    count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        count += 1
        if count % 3 != 0:
            continue  # Skip some frames for speed (optional)

        frame = cv2.resize(frame, (1018, 600))
        results = model_yolo(frame, verbose=False)
        annotated_frame = results[0].plot(boxes=False)

        for r in results:
            bound_box = r.boxes.xyxy
            conf = r.boxes.conf.tolist()
            keypoints = r.keypoints.xyn.tolist()

            print(f'Frame {frame_tot}: Detected {len(bound_box)} bounding boxes')

            for index, box in enumerate(bound_box):
                if conf[index] > 0.55:
                    x1, y1, x2, y2 = box.tolist()

                    data = {}
                    for j in range(len(keypoints[index])):
                        data[f'x{j}'] = keypoints[index][j][0]
                        data[f'y{j}'] = keypoints[index][j][1]

                    df = pd.DataFrame(data, index=[0])
                    dmatrix = xgb.DMatrix(df)

                    pred_probs = model.predict(dmatrix)  # Shape: (1, num_classes)
                    class_idx = int(np.argmax(pred_probs[0]))
                    class_label = label_decoder[class_idx]
                    confidence = float(np.max(pred_probs[0]))

                    print(f'Prediction: {class_label} (Confidence: {confidence:.2f})')

                    # Set color and draw annotation
                    color = (0, 255, 0)  # Green for Normal
                    if class_label != "Normal":
                        color = (0, 0, 255)  # Red for suspicious actions

                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cvzone.putTextRect(
                        annotated_frame,
                        f"{class_label} ({confidence:.2f})",
                        (int(x1), int(y1) + 50),
                        scale=1,
                        thickness=1
                    )

        frame_tot += 1
        cv2.imshow('Frame', annotated_frame)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the function
detect_action(video_path)
