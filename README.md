# CV-Badminton-Matches
AI CUP 2023 - Teaching Computers to Watch Badminton Matches

## Model Architecture
> ![image](https://raw.githubusercontent.com/aisu-programming/CV-Badminton-Matches/main/architecture.png)

## Install
1. Run install_pytorch.sh or install the version you like
2. Run install_mm.sh or manually install [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMPose](https://github.com/open-mmlab/mmpose)
3. Run install_tracknetv2.sh or manually install [TrackNetv2](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2)

## Training Procedure
1. Background Extraction (1_background_extraction.py)
2. Background Clustering (2_background_clustering.py)
3. Ball Detection (3_ball_processing.py)
4. Player Pose Detection (4_pose_detection_*.py)
5. Train models for each columns (5_train_*.py)

## Predicting Procedure
Predict answers of each columns by each models (6_predict_*.py) **step by step**

## Ideas
### Court Detection
(turned out to be unnecessary)
1. Edge detection (Canny): X
2. Homography: X
3. DIY: O

### Background Extraction
1. DIY - "average_method": X
2. DIY - "mode_method": X
3. DIY - "average_method_with_masked_players": O

### Ball Detection
1. Blob detection: X
2. [TrackNetv2](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2): X
3. [TrackNetv2](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2) + postprocess: O

### Player Pose Detection
(Backgound Extraction finished needed)
- [MMDetection](https://github.com/open-mmlab/mmdetection) + [MMPose](https://github.com/open-mmlab/mmpose)

## Others
More content in [my report](https://github.com/aisu-programming/CV-Badminton-Matches/blob/main/report/AI%20CUP%20%E7%AB%B6%E8%B3%BD%E5%A0%B1%E5%91%8A%E8%88%87%E7%A8%8B%E5%BC%8F%E7%A2%BC%EF%BC%8FTEAM_2956%EF%BC%8F%E3%80%8C%E6%95%99%E9%9B%BB%E8%85%A6%E7%9C%8B%E7%BE%BD%E7%90%83%E3%80%8D%E7%AB%B6%E8%B3%BD.pdf) but in Chinese version
