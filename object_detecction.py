##for live detection

# import cv2
# import torch
# import pyttsx3

# # Load YOLOv5 model from the official repository (online mode)
# model = torch.hub.load('ultralytics/yolov5:master', 'yolov5x')  # Load YOLOv5x model from the master branch
# model.eval()  # Set the model to evaluation mode

# cap = cv2.VideoCapture(0)  # Use 0 for default camera, 1 for an external camera

# engine = pyttsx3.init()  # Initialize the TTS engine
# previous_labels = set()  # Set to store previously detected labels

# while True:
#     ret, frame = cap.read()  # Read a frame from the camera

#     # Perform object detection
#     results = model(frame)

#     # Detected objects with their bounding boxes and labels
#     for obj in results.pred[0]:
#         x_min, y_min, x_max, y_max, confidence, label = obj.int().tolist()
#         label_name = model.names[int(label)]
        
#         # Draw bounding box and label on the frame
#         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#         cv2.putText(frame, f'{label_name}: {confidence:.2f}', (x_min, y_min - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
#         # Speak the label name only if it's a new label
#         if label_name not in previous_labels:
#             engine.say(f"I see a {label_name}")
#             engine.runAndWait()
#             previous_labels.add(label_name)  # Add the label to the set to indicate it has been spoken

#     # Display the frame
#     cv2.imshow('Object Detection', frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()  # Release the camera
# cv2.destroyAllWindows()  # Close OpenCV windows

# import cv2
# import torch
# from gtts import gTTS
# import os

# # Load YOLOv5 model from the official repository (online mode)
# model = torch.hub.load('ultralytics/yolov5:master', 'yolov5x')  # Load YOLOv5x model from the master branch
# model.eval()  # Set the model to evaluation mode

# # Read the input image
# image_path = '/home/vansh/Downloads/image_2.jpg'
# original_frame = cv2.imread(image_path)

# # Resize the original frame
# original_frame_resized = cv2.resize(original_frame, (800, 600))  # Adjust the width and height as needed

# # Perform object detection on the resized frame
# resized_frame = cv2.resize(original_frame, (800, 600))  # Resize the frame for detection
# results = model(resized_frame)

# # Detected objects with their bounding boxes and labels
# detected_labels = set()  # Set to store detected labels

# for obj in results.pred[0]:
#     x_min, y_min, x_max, y_max, confidence, label = obj.int().tolist()
#     label_name = model.names[int(label)]
#     detected_labels.add(label_name)  # Add the label to the set to indicate it has been detected
    
#     # Draw bounding box and label on the frame
#     cv2.rectangle(resized_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#     cv2.putText(resized_frame, f'{label_name}: {confidence:.2f}', (x_min, y_min - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# # Convert the set of detected labels to individual sentences
# for label in detected_labels:
#     tts_message = f"I see a {label}"
#     engine = gTTS(tts_message, lang='en')
#     engine.save('temp.mp3')  # Save the speech to a temporary file
#     os.system('mpg321 temp.mp3')  # Play the temporary file using mpg321 player

# # Display the original frame with bounding boxes
# cv2.imshow('Object Detection', resized_frame)
# cv2.imshow('Original Image', original_frame_resized)
# cv2.waitKey(0)  # Wait for a key event
# cv2.destroyAllWindows()  # Close OpenCV windows

                                                                                                                                                                 