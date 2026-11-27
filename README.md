Holistic model is the program to represent the landmarks based on holistic model from mediapipe and CV2
Selfie with gesture  program  takes a selfie based on gesture of palm and save in the current directory

######################################################
Requirements (Python 3.11, mediapipe is pinned to 0.10.21 - newer releases removed the mp.solutions API used here):
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

Run (press q in the window to quit):
.venv\Scripts\python.exe holistic_model.py
.venv\Scripts\python.exe selfie_with_gesture.py

Tests (no webcam needed - the camera and windows are faked):
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest tests

Libs folder contains coordinates calculator and palm position checking medthods based on landmarks from Holistic model

HandLandmark.WRIST 0
HandLandmark.THUMB_CMC 1
HandLandmark.THUMB_MCP 2
HandLandmark.THUMB_IP 3
HandLandmark.THUMB_TIP 4
HandLandmark.INDEX_FINGER_MCP 5
HandLandmark.INDEX_FINGER_PIP 6
HandLandmark.INDEX_FINGER_DIP 7
HandLandmark.INDEX_FINGER_TIP 8
HandLandmark.MIDDLE_FINGER_MCP 9
HandLandmark.MIDDLE_FINGER_PIP 10
HandLandmark.MIDDLE_FINGER_DIP 11
HandLandmark.MIDDLE_FINGER_TIP 12
HandLandmark.RING_FINGER_MCP 13
HandLandmark.RING_FINGER_PIP 14
HandLandmark.RING_FINGER_DIP 15
HandLandmark.RING_FINGER_TIP 16
HandLandmark.PINKY_MCP 17
HandLandmark.PINKY_PIP 18
HandLandmark.PINKY_DIP 19
HandLandmark.PINKY_TIP 20



