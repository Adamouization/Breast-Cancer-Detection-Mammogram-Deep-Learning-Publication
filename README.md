# A Divide and Conquor Approach to Maximise Deep Learning Mammography Classification Accuracies - Published in PLOS ONE [![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg?style=for-the-badge)](https://opensource.org/licenses/BSD-2-Clause) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)


**Publication repository of the "_A Divide and Conquor Approach to Maximise Deep Learning Mammography Classification Accuracies_" peer-reviewed paper published in PLOS ONE.** You can read the paper here: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0280841

## Abstract

Breast cancer claims 11,400 lives on average every year in the UK, making it one of the deadliest diseases. Mammography is the gold standard for detecting early signs of breast cancer, which can help cure the disease during its early stages. However, incorrect mammography diagnoses are common and may harm patients through unnecessary treatments and operations (or a lack of treatments). Therefore, systems that can learn to detect breast cancer on their own could help reduce the number of incorrect interpretations and missed cases. Various deep learning techniques, which can be used to implement a system that learns how to detect instances of breast cancer in mammograms, are explored throughout this paper.

Convolution Neural Networks (CNNs) are used as part of a pipeline based on deep learning techniques. A divide and conquer approach is followed to analyse the effects on performance and efficiency when utilising diverse deep learning techniques such as varying network architectures (VGG19, ResNet50, InceptionV3, DenseNet121, MobileNetV2), class weights, input sizes, image ratios, pre-processing techniques, transfer learning, dropout rates, and types of mammogram projections.

![CNN Model](https://i.imgur.com/dIfhxyz.png)

Multiple techniques are found to provide accuracy gains relative to a general baseline (VGG19 model using uncropped 512x512 pixels input images with a dropout rate of 0.2 and a learning rate of 1×10^−3) on the Curated Breast Imaging Subset of DDSM (CBIS-DDSM) dataset. These techniques involve transfer learning pre-trained ImagetNet weights to a MobileNetV2 architecture, with pre-trained weights from a binarised version of the mini Mammography Image Analysis Society (mini-MIAS) dataset applied to the fully connected layers of the model, coupled with using weights to alleviate class imbalance, and splitting CBIS-DDSM samples between images of masses and calcifications. Using these techniques, a 5.28% gain in accuracy over the baseline model was accomplished. Other deep learning techniques from the divide and conquer approach, such as larger image sizes, do not yield increased accuracies without the use of image pre-processing techniques such as Gaussian filtering, histogram equalisation and input cropping.

## Citation

### Code citation (this GitHub repository) [![DOI](https://zenodo.org/badge/345135430.svg)](https://zenodo.org/badge/latestdoi/345135430)
```
@software{Jaamour_Adamouization_Breast-Cancer-Detection-Mammogram-Deep-Learning-Publication_PLOS_ONE_2023,
    author = {Jaamour, Adam and Myles, Craig},
    license = {BSD-2-Clause},
    month = may,
    title = {{Adamouization/Breast-Cancer-Detection-Mammogram-Deep-Learning-Publication: PLOS ONE Submission}},
    url = {https://github.com/Adamouization/Breast-Cancer-Detection-Mammogram-Deep-Learning-Publication},
    version = {1.2},
    year = {2023}
}
```

### Published paper citation (PLOS ONE)
```
@article{10.1371/journal.pone.0280841,
    doi = {10.1371/journal.pone.0280841},
    author = {Jaamour, Adam AND Myles, Craig AND Patel, Ashay AND Chen, Shuen-Jen AND McMillan, Lewis AND Harris-Birtill, David},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {A divide and conquer approach to maximise deep learning mammography classification accuracies},
    year = {2023},
    month = {05},
    volume = {18},
    url = {https://doi.org/10.1371/journal.pone.0280841},
    pages = {1-24},
    number = {5},
}
```

## Environment setup and usage

Clone the repository:

```
git clone https://github.com/Adamouization/Breast-Cancer-Detection-Mammogram-Deep-Learning-Publication
cd Breast-Cancer-Detection-Mammogram-Deep-Learning-Publication
```

Create a virtual conda environment:

```
conda create -n mammography python=3.6.13
conda activate mammography
```

Install requirements:
```
pip install -r requirements.txt
```

Create `output`and `save_models` directories to store the results:

```
mkdir output
mkdir saved_models
```

`cd` into the `src` directory and run the code:

```
cd ./src
main.py [-h] -d DATASET [-mt MAMMOGRAMTYPE] -m MODEL [-r RUNMODE] [-lr LEARNING_RATE] [-b BATCHSIZE] [-e1 MAX_EPOCH_FROZEN] [-e2 MAX_EPOCH_UNFROZEN] [-roi] [-v] [-n NAME]
```

where:
* `-h` is a flag for help on how to run the code.
* `DATASET` is the dataset to use. Must be either `mini-MIAS`, `mini-MIAS-binary` or `CBIS-DDMS`. Defaults to `CBIS-DDMS`.
* `MAMMOGRAMTYPE` is the type of mammograms to use. Can be either `calc`, `mass` or `all`. Defaults to `all`.
* `MODEL` is the model to use. Must be either `VGG-common`, `VGG`, `ResNet`, `Inception`, `DenseNet`, `MobileNet` or `CNN`.
* `RUNMODE` is the mode to run in (`train` or `test`). Default value is `train`.
* `LEARNING_RATE` is the optimiser's initial learning rate when training the model during the first training phase (frozen layers). Defaults to `0.001`. Must be a positive float.
* `BATCHSIZE` is the batch size to use when training the model. Defaults to `2`. Must be a positive integer.
* `MAX_EPOCH_FROZEN` is the maximum number of epochs in the first training phrase (with frozen layers). Defaults to `100`.
* `MAX_EPOCH_UNFROZEN`is the maximum number of epochs in the second training phrase (with unfrozen layers). Defaults to `50`.
* `-roi` is a flag to use versions of the images cropped around the ROI. Only usable with mini-MIAS dataset. Defaults to `False`.
* `-v` is a flag controlling verbose mode, which prints additional statements for debugging purposes.
* `NAME` is name of the experiment being tested (used for saving plots and model weights). Defaults to an empty string.

## For best model described in paper:
```
python main.py -d CBIS-DDSM -mt all -m MobileNet -r train -lr 0.0001
python main.py -d CBIS-DDSM -mt all -m MobileNet -r test -lr 0.0001
```

## Dataset installation

#### DDSM and CBIS-DDSM datasets

These datasets are very large (exceeding 160GB) and more complex than the mini-MIAS dataset to use. They were downloaded by the University of St Andrews School of Computer Science computing officers onto \textit{BigTMP}, a 15TB filesystem that is mounted on the Centos 7 computer lab clients with NVIDIA GPUsusually used for storing large working data sets. Therefore, the download process of these datasets will not be covered in these instructions.

The generated CSV files to use these datasets can be found in the `/data/CBIS-DDSM` directory, but the mammograms will have to be downloaded separately. The DDSM dataset can be downloaded [here](http://www.eng.usf.edu/cvprg/Mammography/Database.html), while the CBIS-DDSM dataset can be downloaded [here](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM#5e40bd1f79d64f04b40cac57ceca9272).

#### mini-MIAS dataset

* This example will use the [mini-MIAS](http://peipa.essex.ac.uk/info/mias.html) dataset. After cloning the project, travel to the `data/mini-MIAS` directory (there should be 3 files in it).

* Create `images_original` and `images_processed` directories in this directory: 

```
cd data/mini-MIAS/
mkdir images_original
mkdir images_processed
```

* Move to the `images_original` directory and download the raw un-processed images:

```
cd images_original
wget http://peipa.essex.ac.uk/pix/mias/all-mias.tar.gz
```

* Unzip the dataset then delete all non-image files:

```
tar xvzf all-mias.tar.gz
rm -rf *.txt 
rm -rf README 
```

* Move back up one level and move to the `images_processed` directory. Create 3 new directories there (`benign_cases`, `malignant_cases` and `normal_cases`):

```
cd ../images_processed
mkdir benign_cases
mkdir malignant_cases
mkdir normal_cases
```

* Now run the python script for processing the dataset and render it usable with Tensorflow and Keras:

```
python3 ../../../src/dataset_processing_scripts/mini-MIAS-initial-pre-processing.py
```

## License 
* see [BSD 2-Clause License](https://github.com/Adamouization/Breast-Cancer-Detection-Mammogram-Deep-Learning-Publication/blob/master/LICENSE) file.

## Authors

* Adam Jaamour (adam[at]jaamour[dot]com)
* Craig Myles
* Ashay Patel
* Shuen-Jen Chen
* Lewis McMillan
* David Harris-Birtill
