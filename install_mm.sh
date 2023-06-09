pip install ffmpegcv openmim
mim install mmcv-full

git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
cd ..

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
cd ..

# git clone https://github.com/open-mmlab/mmediting.git
# cd mmediting
# pip3 install -e .
# cd ..

# pip install git+https://github.com/votchallenge/toolkit.git
# git clone https://github.com/open-mmlab/mmtracking.git
# cd mmtracking
# pip install -r requirements/build.txt
# pip install -v -e .
# cd ..

pip install mmflow mmengine

git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -v -e .
cd ..