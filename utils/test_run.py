from optimized_action_inference import detect_action

input_video = "./action_recog_backend/uploads/segment_11.mp4"
output_video = "./action_recog_backend/output/out_seg_part_11.mp4"

detect_action(input_video, output_video)

print("Successfully completed the inference. output saved to :", output_video)