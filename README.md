# denoising image

 

Pytorch Code for the paper ["Controllable Deep Learning Denoising Model for Ultrasound Images Using Synthetic Noisy Image"](CGI2023)
# Requirements

python=3.6  
1.4.0<=pytorch<=1.7.0



# Inroduction

Medical ultrasound imaging has gained widespread prevalence in human muscle and internal organ diagnosis. Nevertheless, various factors such as the interference effect of ultrasonic echoes, mutual interference between scattered beams, inhomogeneity and uncertainty in the spatial distribution of human body tissue, inappropriate operation, and imaging signal transmission processes, can lead to noise and distortion in ultrasound images. These fac-tors make it difficult to obtain clean and accurate ultrasound images, which may adversely affect medical diagnosis and treatment processes. While tradi-tional denoising methods are time-consuming, they are also not effective in removing speckle noise while retaining image details, leading to potential misdiagnosis. Therefore, there is a significant need to accurately and quickly denoise medical ultrasound images to enhance image quality. In this paper, we propose a flexible and lightweight deep learning denoising method for ul-trasound images. Initially, we utilize a considerable number of natural imag-es to train the convolutional neural network for acquiring a pre-trained de-noising model. Next, we employ the plane-wave imaging technique to gen-erate simulated noisy ultrasound images for further transfer learning of the pre-trained model. As a result, we obtain a non-blind, lightweight, fast, and accurate denoiser. Experimental results demonstrate the superiority of our proposed method in terms of denoising speed, flexibility, and effectiveness compared to conventional convolutional neural network denoisers for ultra-sound images.