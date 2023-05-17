# CV-Badminton-Matches
AI CUP 2023 - Teaching Computers to Watch Badminton Matches

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

## Training Procedure
1. Background Extraction (1_background_extraction.py)
2. Background Clustering (2_background_clustering.py)
3. Ball Detection (3_ball_processing.py)
4. Player Pose Detection (4_pose_detection_*.py)
5. Train models for each columns (5_train_*.py)

## Predicting Procedure
1. Predict answers of each columns by each models (6_predict_*.py)
