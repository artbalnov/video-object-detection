#!/bin/bash
rm /usr/lib/python3.11/EXTERNALLY-MANAGED

apt update
apt install python3-pip -y

pip install opencv-python
pip install mediapipe
pip install ultralytics[export]