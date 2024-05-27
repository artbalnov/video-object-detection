# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main scripts to run object detection."""

import argparse
import time

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ultralytics import YOLO

from utils import visualize

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()


def run(model_path: str, max_results: int, score_threshold: float, 
        vido_file: str, camera_id: int, width: int, height: int, max_fps: int) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    max_results: Max number of detection results.
    score_threshold: The score threshold of detection results.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
  """

  # Start capturing video input from the camera
  if camera_id == -1:
    cap = cv2.VideoCapture(vido_file)
  else: 
    cap = cv2.VideoCapture(camera_id)
  
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 50  # pixels
  left_margin = 24  # pixels
  text_color = (255, 255, 255)  # black
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  detection_frame = None
  detection_result_list = []

  def detection_callback(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
      detection_result_list.append(result)
      calc_fps()

  def calc_fps():
      global FPS, COUNTER, START_TIME

      # Calculate the FPS
      if COUNTER % fps_avg_frame_count == 0:
          FPS = fps_avg_frame_count / (time.time() - START_TIME)
          START_TIME = time.time()

      COUNTER += 1
      
  detector: vision.ObjectDetector 
  # Initialize the object detection model
  if "yolo" in model_path:  
     model = YOLO("models/yolov8n.pt")
  else:
     base_options = python.BaseOptions(model_asset_path=model_path)
     options = vision.ObjectDetectorOptions(base_options=base_options,
                                         running_mode=vision.RunningMode.LIVE_STREAM,
                                         max_results=max_results, score_threshold=score_threshold,
                                         result_callback=detection_callback)
     detector = vision.ObjectDetector.create_from_options(options)
    
  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()

    now = time.time()
    
    # image=cv2.resize(image,(640,480))
    # if not success:
    #   sys.exit(
    #       'ERROR: Unable to read from webcam. Please verify your webcam settings.'
    #   )
      
    image = cv2.flip(image, 1)

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(FPS)
    text_location = (left_margin, row_size)
    current_frame = image
    cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                font_size, text_color, font_thickness, cv2.LINE_AA)
    
    if "yolo" in model_path:
        # Run YOLOv8 inference on the frame
        results = model(image, imgsz=[height,width], conf=score_threshold, max_det=max_results)
        detection_frame = results[0].plot()
        calc_fps()
    else:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        # Run object detection using the model.   
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        if detection_result_list:
            current_frame = visualize(current_frame, detection_result_list[0])
            detection_frame = current_frame
            detection_result_list.clear()

    if detection_frame is not None:
        cv2.imshow('object_detection', detection_frame)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    
    timeDiff = time.time() - now
    if (timeDiff < 1.0/(max_fps)): time.sleep( 1.0/(max_fps) - timeDiff )

  if detector !=None: 
    detector.close()

  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='models/ssd_mobilenet_v1.tflite')
#      default='models/efficientdet_lite0.tflite')
  parser.add_argument(
      '--maxResults',
      help='Max number of detection results.',
      required=False,
      default=3)
  parser.add_argument(
      '--scoreThreshold',
      help='The score threshold of detection results.',
      required=False,
      type=float,
      default=0.30)
  # Finding the camera ID can be very reliant on platform-dependent methods. 
  # One common approach is to use the fact that camera IDs are usually indexed sequentially by the OS, starting from 0. 
  # Here, we use OpenCV and create a VideoCapture object for each potential ID with 'cap = cv2.VideoCapture(i)'.
  # If 'cap' is None or not 'cap.isOpened()', it indicates the camera ID is not available.
  parser.add_argument(
      '--cameraId', help='camera ID', required=False, type=int, default=-1)
  parser.add_argument(
      '--videoFile', help='Path to thevideo file', required=False, type=str, default='data/James.mp4')
  parser.add_argument(
      '--maxFPS', help='Max FPS, for video file use video FPS', required=False, type=int, default=30)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  args = parser.parse_args()

  run(args.model, int(args.maxResults),
      args.scoreThreshold, args.videoFile, int(args.cameraId), args.frameWidth, args.frameHeight, args.maxFPS)


if __name__ == '__main__':
  main()
