# GAN Path Finder
### Generative  Adversarial  Networks  for  Path  Planning  in  2D
This repository is a fork of the original author's repository (https://github.com/PathPlanning/GAN-Path-Finder)


## Pre-requisite
Please ensure that a conda environment is set up with the following libraries. Also, please ensure that the conda environment is activated first before runnignt he code.
- OpenCV
- Numpy
- Matplotlib
- CUDA Toolkit (version 12.6 is used)
- PyTorch
- scipy
- scikit-image
- jupyter
- tqdm
- Pillow
- os
- csv
- ipyparallel
- trimesh
- glob2
- re
- h5py
- PyYAML
- joblib
- tensorboardX
- pylzma
- k3d
- randomcolor


#### Generate Dataset
Run the `DatasetGeneration.py` script to generate the dataset.


#### Fine-tuning dataset location
The fie-tuning dataset is located at `./data/size_64/20_den3` and the test cases are located at `./data/size_64/ideal`


#### Model training (Provided by the author's README)
```
python ./train.py --img_size 64 --channels 1 --num_classes 3 --dataset_dir ./size_64/20_den/ --results_dir ./size_64/20_den/results/ --batch_size 32 --epoch_count 1 --number_of_epochs 100
```
Change parameters to change the training: 
+ img_size - size of the input and output images/field_size of the grid.
+ channels - number of channels in the input image (in case you want to change input images to accommodate multi-channel input).
+ num_classes - output number of channels/classes (3 classes: free space, obstacles and path pixels).
+ dataset_dir - path to the dataset with images.
+ results_dir - where all the results/model weights will be saved.
+ batch_size - batch size for training.
+ epoch_count - from which epoch to start (1 to start from the beginning, any other epoch to continue training from mentioned epoch and last checkpoint in results folder).
+ number_of_epochs - number of epochs to train.


#### Model Prediction
To get the GAN-Path-Finder model prediction, run the `val_results.py` python script.
Please ensure that the prediction data / test cases are included in the file directory shown in the script. 


#### A* with Dynamic Cost Solution
To get the A* solution, run the `ResultComparison.py` python script
Please ensure that the input image is included in the root folder of the current working directory.
Additionally, the solution can only calculate the solution for one input at a time. Please use the counter variable in the script to change the iteration.


#### Results Comparison
To evaluate the GAN-Path-Finder's prediction and the A* with dynamic cost's solution, please run the `ResultComparison.py` script.
Please ensure that the test-cases are within the root folder of the current working directory.
