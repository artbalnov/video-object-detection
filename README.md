To setup all dependencies:
```
sudo ./setup.sh
```

To get all available args: 
```
python3 detect.py --help
```

To run obeject detecion:
```
python3 detect.py --maxResults 3 --scoreThreshold 0.3 --model models/ssd_mobilenet_v1.tflite
```
