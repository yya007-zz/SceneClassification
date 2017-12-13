# Scene Classification using Object Detection

## Introduction
Scene classification is a challenging but important image classification task. Variation in direction, weather, time and subject recognition directly complicate the process. Historically, deep learning is rapidly overtaking classical approaches for scene classification. However, the deep learning approach needs a large number of images. [Zhou et al.](http://places2.csail.mit.edu/PAMI_places.pdf) shows that the Alexnet and successful deep learning approach in general image classification actually achieve similar performance with large scene image dataest. In this project, we does scene classification by only using a small amount of images.

The data used here is from [Miniplaces Challenge](https://github.com/CSAILVision/miniplaces). The goal of that challenge is to identify the scene category depicted in a photograph. The origin data is from the [Places2 dataset](http://places2.csail.mit.edu/), which has 10+ million images and 400+ unique scene categories. The data used in this project only has 100 scene categories and 100,000 images for training, 9,000 images for validation, and 1,000 images for testing.


We divide all data into three dataset.
```
               Number of images
   Dataset     TRAIN        VALIDATION   TEST
  ------------------------------------------------
   MiniPlaces  100,000      9,000       1,000
```
Database statistics is as follows:

```
  Training:   100,000 images, with 1000 images per category
  Validation: 10,000 images, with 100 images per category
  Test:   10,000 images, with 100 images per category
```

## Dependencies
* Python 2.6 or newer 
* numpy
```
conda install numpy
```
* scipy
```
conda install scipy
```
* Tensorflow
```
conda install tensorflow
```
For GPU user:
```
conda install tensorflow-gpu
```
* matplotlib
```
conda install matplotlib
```
* opencv
```
conda install -c menpo opencv
```
## Get Data
We already packaged all the data you need to download to a script. Simple run follwoing command to get all data.
```
bash get_data.sh
```
## Data Preprocessing
In order to run the network, we need to preprocess the data in order to standardize them.
```
python data_processsor.py
```
## Run
We store common parameters combination inside exp.py. In order to run a the model with parameters combination exp1 inside exp.py. Simply run following command.
```
python main.py exp1
```
## Parameters
The real meaning of each parameters are shown following:
```
  Training:   100,000 images, with 1000 images per category
  Validation: 10,000 images, with 100 images per category
  Test:   10,000 images, with 100 images per category
```

## Reference 
* [Miniplaces Challenge](https://github.com/CSAILVision/miniplaces)
* [Places1 Database](http://places.csail.mit.edu)
* [Places2 Database](http://places2.csail.mit.edu)

## Contact
* [Yuang Yao](yuangyao@mit.edu), [Hung-Jui Huang](joehuang@mit.edu)
* If you an MIT student taking 6.867 or 6.819/6.869: Advances in Computer Vision, please contact the teaching staff for any course-related problems.
