from action_recog_backend.utils.optimized_action_inference import detect_action

input_video = "./action_recog_backend/uploads/output_part_4.mp4"
output_video = "./action_recog_backend/output/output_part_4.1.mp4"

detect_action(input_video, output_video)

print("Successfully completed the inference. output saved to :", output_video)