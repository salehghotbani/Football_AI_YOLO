# Detect and track players, ball and referee in football match using `YOLO v11`, `roboflow` and `supervision` packages.

- Using `roboflow` to create dataset
- Using `YOLO` to train model and detect players and ball
- Using `supervision` to visualize boxes around detected objects and set label for them

---

### Install pytorch compatible with `CUDA`

#### First of all download and install
`CUDA toolkit` compatible with your Nvidia graphics card from [this link](https://developer.nvidia.com/cuda-downloads)

#### Then install `pytorch` from [this way](https://pytorch.org/get-started/locally/):

![installing pytorch compatible to CUDA platform](https://storage4.fastupload.io/cache/plugins/filepreviewer/1052705/8017b8ce4f4a0e776cea0e8a587c715f726898722aa25e17f183bb5bd0173e56/1100x800_cropped.jpg)

```commandline
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

### Verify Nvidia is access

```commandline
nvidia-smi
```

-----

### install required packages

```commandline
pip install -q ultralytics roboflow supervision
```

---

### Download ball, players and referee detection dataset from [this link](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)

![installing pytorch compatible to CUDA platform](https://storage4.fastupload.io/cache/plugins/filepreviewer/1052707/653f60f51208aca0f168a58b3ec75bcef3dd428b26861d9dc37b3e69f7566bca/1100x800_cropped.jpg)

---

### Download base model of `YOLO` form [this link](https://docs.ultralytics.com/tasks/detect/#models)

![models image](https://storage6.fastupload.io/cache/plugins/filepreviewer/1052706/9695a0099b9f2d20a6aec3a756c7bcd120848585c9bf90eb33e72a69599ef4ed/1100x800_cropped.jpg)

#### I used `YOLO11s`

---

### Imports

``` python
import os
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm
from sports.common.team import TeamClassifier
import supervision as sv
```

---

### Predefined Variables

```python
HOME = os.getcwd()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

---

### Train model to detect `players`, `ball` and `referee`

```python
base_model = YOLO(f'{HOME}/datasets/yolo11s.pt')
base_model.train(data=f"{HOME}/football/data.yaml", epochs=50, batch=8, imgsz=1280, plots=True, device=DEVICE)
```

--------

### Validate trained model

```python
base_model.val(data=f'{HOME}/football/data.yaml', imgsz=1280, device=DEVICE)
```

### Our Model TRained Path

```python
OUR_MODEL_PATH = f'{HOME}/runs/detect/train4/weights/best.pt'
SOURCE_VIDEO_PATH = f"{HOME}\clips\download2.mp4"
```

---

### Download football match clips

- https://drive.google.com/uc?id=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF
- https://drive.google.com/uc?id=19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf
- https://drive.google.com/uc?id=1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-
- https://drive.google.com/uc?id=1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU
- https://drive.google.com/uc?id=1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu

---

### Object IDs

```python
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3
```

---

### Detect `players`, `ball` and `referee` just using `YOLO`

```python
model = YOLO(OUR_MODEL_PATH)
result = model(source=SOURCE_VIDEO_PATH, show=True, conf=0.4, save=True)
```

---

### Collect Crops Of Players

```python
STRIDE = 30


def extract_crops(source_video_path: str):
    frame_generator = sv.get_video_frames_generator(source_video_path, stride=STRIDE)
    yolo_model = YOLO(OUR_MODEL_PATH)

    crops = []
    for frame in tqdm(frame_generator, desc="Extracting crops..."):
        results = yolo_model(frame, conf=0.3, save=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == PLAYER_ID]
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        crops += [
            sv.crop_image(frame, xyxy)
            for xyxy in detections.xyxy
        ]
    return crops
```

```python
crops = extract_crops(SOURCE_VIDEO_PATH)
len(crops)
sv.plot_images_grid(crops[:100], grid_size=(10, 10))
```

---

### Team Classification

```commandline
pip install -q git+https://github.com/roboflow/sports.git
```

```python
def resolve_goalkeepers_team_id(players: sv.Detections, goalkeepers: sv.Detections):
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_id)
```

```python
crops = extract_crops(SOURCE_VIDEO_PATH)
team_classifier = TeamClassifier(device=DEVICE)
team_classifier.fit(crops)

yolo_model = YOLO(OUR_MODEL_PATH)

ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
)
label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    text_color=sv.Color.from_hex('#000000'),
    text_position=sv.Position.BOTTOM_CENTER
)
triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FFD700'),
    base=20,
    height=17,
)

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(frame_generator)

results = yolo_model(frame, conf=0.4, save=False)[0]

detections = sv.Detections.from_ultralytics(results)

ball_detections = detections[detections.class_id == BALL_ID]
ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

all_detections = detections[detections.class_id != BALL_ID]
all_detections = all_detections.with_nmm(threshold=0.5, class_agnostic=True)

players_detections = all_detections[all_detections.class_id == PLAYER_ID]
goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
players_detections.class_id = team_classifier.predict(players_crops)

goalkeepers_detections.class_id = resolve_goalkeepers_team_id(players_detections, goalkeepers_detections)
referees_detections.class_id -= 1

all_detections = sv.Detections.merge([players_detections, goalkeepers_detections, referees_detections])

annotated_frame = frame.copy()
annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=all_detections)
annotated_frame = triangle_annotator.annotate(scene=annotated_frame, detections=ball_detections)
sv.plot_image(annotated_frame)
```

![Sample result](https://storage4.fastupload.io/cache/plugins/filepreviewer/1052776/1a7ad703ff7d996a3bf72ce6d5f6ad63e94c7ab18c6a17c93bf8dc32ddece3e9/1100x800_cropped.jpg)

```python
crops = extract_crops(SOURCE_VIDEO_PATH)
team_classifier = TeamClassifier(device=DEVICE)
team_classifier.fit(crops)

yolo_model = YOLO(OUR_MODEL_PATH)

ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
)
label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    text_color=sv.Color.from_hex('#000000'),
    text_position=sv.Position.BOTTOM_CENTER
)
triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FFD700'),
    base=20,
    height=17,
)

TARGET_VIDEO_PATH = f"{HOME}/result_clips/result.mp4"

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
video_sink = sv.VideoSink(TARGET_VIDEO_PATH, video_info=video_info)
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

with video_sink:
    for frame in tqdm(frame_generator, total=video_info.total_frames):
        results = yolo_model(frame, conf=0.4, save=False, device=DEVICE)[0]

        detections = sv.Detections.from_ultralytics(results)

        ball_detections = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        all_detections = detections[detections.class_id != BALL_ID]
        all_detections = all_detections.with_nmm(threshold=0.5, class_agnostic=True)

        players_detections = all_detections[all_detections.class_id == PLAYER_ID]
        goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
        referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        players_detections.class_id = team_classifier.predict(players_crops)

        goalkeepers_detections.class_id = resolve_goalkeepers_team_id(players_detections, goalkeepers_detections)
        referees_detections.class_id -= 1

        all_detections = sv.Detections.merge([players_detections, goalkeepers_detections, referees_detections])

        annotated_frame = frame.copy()
        annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=all_detections)
        annotated_frame = triangle_annotator.annotate(scene=annotated_frame, detections=ball_detections)

        video_sink.write_frame(annotated_frame)
```

[![Result video](https://storage4.fastupload.io/cache/plugins/filepreviewer/1052776/1a7ad703ff7d996a3bf72ce6d5f6ad63e94c7ab18c6a17c93bf8dc32ddece3e9/1100x800_cropped.jpg)](https://github.com/salehghotbani/Football_Yolo11_Supervision_Roboflow/blob/main/result_clips/result.mp4)
