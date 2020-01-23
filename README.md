## Emotion Recognition Using OpenVINO Toolkit
Emotion recognitoion based on pretrained model from OpenVINO `emotion-recognition-retail-003`.

### Dependencies
1. Python 3.7
2. OpenVINO toolkit
3. OpenCV 4.x.x
4. Numpy

## How to Use

```
python app.py -m "pretrained model path" -c "opencv haar cascade model"
```

### Hadwawere
By default this code use MYRIAD. CPU and GPU support will be updated soon.

### TODOs

1. Support multiple hardware
2. Support video input
3. Detect only main face
