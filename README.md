# denoising image

 

Pytorch Code for the paper ["Controllable Deep Learning Denoising Model for Ultrasound Images Using Synthetic Noisy Image"](CGI2023)
# Requirements

python=3.6  
1.4.0<=pytorch<=1.7.0
### 1. Testing

If you want to denoise an image using a one of the pretrained models
found under the *models* folder you can execute
```
python test_ffdnet_ipol-anno.py --input input.png --noise_sigma 25 --add_noise True --no_gpu
```
To run the algorithm on CPU instead of GPU:
```
python test_ffdnet_ipol-anno.py \
	--input input.png \
	--noise_sigma 25 \
	--add_noise True \
	--no_gpu
```
**NOTES**
* Models have been trained for values of noise in [0, 75]
* *add_noise* can be set to *False* if the input image is already noisy

### 2. Training

#### Prepare the databases

First, you will need to prepare the dataset composed of patches by executing
*prepare_patches.py* indicating the paths to the directories containing the 
training and validation datasets by passing *--trainset_dir* and
*--valset_dir*, respectively.

Image datasets are not provided with this code, but the following can be downloaded
Training:
[Waterloo Exploration Database](https://ece.uwaterloo.ca/~k29ma/exploration/)
Validation:
[Kodak Lossless True Color Image Suite](http://r0k.us/graphics/kodak/)

**NOTES**
* To prepare a grayscale dataset:

 ```python prepare_patches.py --gray```

* *--max_number_patches* can be used to set the maximum number of patches
contained in the database

#### Train a model

A model can be trained after having built the training and validation databases 
(i.e. *train_rgb.h5* and *val_rgb.h5* for color denoising, and *train_gray.h5*
and *val_gray.h5* for grayscale denoising).
Only training on GPU is supported.
```


python train.py \--model_name FDDNet \--batch_size 128 \--epochs 2 \--noiseIntL 0 75 --val_noiseL 25
python train.py \--batch_size 128 \--epochs 2 \--noiseIntL 0 75 --val_noiseL 25



# Inroduction

Medical ultrasound imaging has gained widespread prevalence in human muscle and internal organ diagnosis. Nevertheless, various factors such as the interference effect of ultrasonic echoes, mutual interference between scattered beams, inhomogeneity and uncertainty in the spatial distribution of human body tissue, inappropriate operation, and imaging signal transmission processes, can lead to noise and distortion in ultrasound images. These fac-tors make it difficult to obtain clean and accurate ultrasound images, which may adversely affect medical diagnosis and treatment processes. While tradi-tional denoising methods are time-consuming, they are also not effective in removing speckle noise while retaining image details, leading to potential misdiagnosis. Therefore, there is a significant need to accurately and quickly denoise medical ultrasound images to enhance image quality. In this paper, we propose a flexible and lightweight deep learning denoising method for ul-trasound images. Initially, we utilize a considerable number of natural imag-es to train the convolutional neural network for acquiring a pre-trained de-noising model. Next, we employ the plane-wave imaging technique to gen-erate simulated noisy ultrasound images for further transfer learning of the pre-trained model. As a result, we obtain a non-blind, lightweight, fast, and accurate denoiser. Experimental results demonstrate the superiority of our proposed method in terms of denoising speed, flexibility, and effectiveness compared to conventional convolutional neural network denoisers for ultra-sound images.
