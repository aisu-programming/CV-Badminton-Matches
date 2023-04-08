# # 2D Human Pose Bottom-Up Video Demo (Slow!!!)
# python mmpose/demo/bottom_up_video_demo.py ^
#        mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py ^
#        https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512-bcb8c247_20200816.pth ^
#        --out-video-root pose_output ^
#        --video-path data/train/00001/00001.mp4

#==================================================#

# 2D Human Pose Top-Down Video Demo
# Single-frames
python mmpose/demo/top_down_video_demo_with_mmdet.py ^
       mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py ^
       https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth ^
       mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288_dark.py ^
       https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth ^
       --out-video-root pose_output ^
       --video-path data/train/00001/00001.mp4

# # Multi-frames (Not good)
# python mmpose/demo/top_down_video_demo_with_mmdet.py ^
#        mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py ^
#        https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth ^
#        mmpose/configs/body/2d_kpt_sview_rgb_vid/posewarper/posetrack18/hrnet_w48_posetrack18_384x288_posewarper_stage2.py ^
#        https://download.openmmlab.com/mmpose/top_down/posewarper/hrnet_w48_posetrack18_384x288_posewarper_stage2-4abf88db_20211130.pth ^
#        --use-multi-frames ^
#        --out-video-root pose_output ^
#        --video-path data/train/00001/00001.mp4

#==================================================#

# 2D Human Whole-Body Pose Top-Down Video Demo
python mmpose/demo/top_down_video_demo_with_mmdet.py ^
       mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py ^
       https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth ^
       mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py ^
       https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth ^
       --out-video-root pose_output ^
       --video-path data/train/00001/00001.mp4

#==================================================#

# Object Detection
# No GPU Acceleration
python mmdetection/demo/video_demo.py data/train/00001/00001.mp4 ^
       mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py ^
       mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth ^
       --out result.mp4

# GPU Acceleration
python mmdetection/demo/video_gpuaccel_demo.py data/train/00001/00001.mp4 ^
       mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py ^
       mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth ^
       --nvdecode --out result.mp4

#==================================================#

# Video Super Resolution
python mmediting/demo/restoration_video_demo.py ^
       mmediting/configs/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_4x2_300k_vimeo90k_bd.py ^
       https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305-ab315ab1.pth ^
       data/train/00001/images/ ^
       ./vsr_output

python mmediting/demo/restoration_video_demo.py ^
       mmediting/configs/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4.py ^
       https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth ^
       data/train/00001/images/ ^
       ./vsr_output

#==================================================#

# Multiple Object Tracking
python mmtracking/demo/demo_mot_vis.py ^
       mmtracking/configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py ^
       --checkpoint https://download.openmmlab.com/mmtracking/mot/bytetrack/bytetrack_yolox_x/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth ^
       --input data/train/00001/00001.mp4 ^
       --output mot.mp4

python demo_mot_vis.py