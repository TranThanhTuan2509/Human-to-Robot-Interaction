# Human-to-Robot Interaction: Learning from Video Demonstration for Robot Imitation

## Prerequisites
Create a conda env called human2robot, install pytorch-1.12.1, cuda-11.3:
```
conda create --name human2robot python=3.8
conda activate human2robot
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
Then, clone the code
```
git clone https://github.com/TranThanhTuan2509/Human-to-Robot-Interaction.git

```
Install all the python dependencies using pip:
```
pip install -r requirements.txt
```
Compile the cuda dependencies using following simple commands:
```
cd lib
python setup.py build develop
```
make sure your `CUDA driver` version is matched with you `nvcc` version if you want to run the model on GPU

## Performance (AP)
### Video understanding
Download both [action recognition checkpoint](https://drive.google.com/file/d/1oZpapQmfzchaC9-GR4uIrawlye-kXaVf/view?usp=drive_link) and [hand detection](https://drive.google.com/open?id=1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE">faster_rcnn_1_8_132028.pth) and save them as follows:

Action recognition checkpoint to `/Human-to-Robot-Interaction/mmaction2/pretrained_file_and_checkpoint`

Hand detection checkpoint to `/Human-to-Robot-Interaction/video_understanding_checkpoint/res101_handobj_100K/pascal_voc`

## Test
To evaluate the understanding performance, run:
```
python3 demo.py --checkepoch=8 --checkpoint=132028 --video video_path
```
Alternatively, you can put a set of videos inside `/Human-to-Robot-Interaction/Dataset/Videos`, and modify your video file names to numbers.
