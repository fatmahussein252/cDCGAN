# cDCGAN

## Overview
This repo contains an implementation of a cDCGAN (Conditional Deep Convolutianl GAN) to be trained on a reduced MNIST dataset to produce
synthetic handwritten digits. The generated data aims to augment training samples, reducing
dependency on real data. The GAN is implemented in Python using TensorFlow/Keras, while the
synthetic data’s effectiveness is evaluated using a modified LeNet-5 architecture. To mitigate
overfitting, dropout layers are added after each fully connected layer. Detailed code explanations
are provided in the notebook file and the LeNet5_utils.py file in this repo.

To generate high-quality, low-noise synthetic images, a Deep Convolutional GAN (DCGAN) were used, following the architectural guidelines proposed by Alec Radford et al for stable training. Additionally, the model were extended to a Conditional DCGAN (cDCGAN) to enable label-guided generation of MNIST digits, to be able to generate class-specific synthetic data. The architecture details are well explained in the notebook file.


## GAN Training Process and Results
The Conditional DCGAN was trained for 50 epochs.
### The average generator and discriminator losses per epoch:
![GAN_losses_50epoch](https://github.com/user-attachments/assets/76e3bcc1-6d1e-4d74-868e-0ec2c4f285c2)

Initially, the discriminator loss (blue) drops
from 0.7 to 0.6 within 5 epochs, indicating it quickly learns to distinguish real and fake images,
while the generator loss (orange) decreases from 1.1 to 0.8, showing improvement in fooling the
discriminator. After ~15 epochs, both losses stabilize at 0.63 and 0.85, respectively, suggesting a
balance where the discriminator remains slightly dominant. Despite this saturation, generated
images continued to improve slightly in quality (e.g., sharper digits), which is typical as GAN
losses don’t directly reflect perceptual quality. Training was continued to 50 epochs to capture
these gradual improvements.
### Generated images after training:
![g](https://github.com/user-attachments/assets/98a8c009-49fd-4942-a9d0-1b3f9f70deec)

## Table of Comparison
The table below shows the training and test accuracies of the LeNet-5 model, trained for 8
epochs on different combinations of real and synthetic data.

| Generated Examples/digit | 300 Real | 700 Real | 1,000 Real |
|--------------------|----------|----------|------------|
| 0 | train acc: 94.9%, test acc: 93.4% | train acc: 96.7%, test acc: 96.1% | train acc: 97.6%, test acc: 97.4% |
| 1000 | train acc: 98.4%, test acc: 95.1% | train acc: 98.1%, test acc: 96.4% | train acc: 98.5%, test acc: 97.7% |
| 2000 | train acc: 99.0%, test acc: 94.7% | train acc: 98.6%, test acc: 96.1% | train acc: 98.6%, test acc: 97.1% |
| 3000 | train acc: 99.4%, test acc: 95.8% | train acc: 99.1%, test acc: 96.4% | train acc: 98.9%, test acc: 96.5% |

## Key Findings
   - GAN-generated data **improves accuracy** when real data is scarce (e.g., +4% for 300 real samples + 3,000 synthetic).  
   - Synthetic data alone cannot fully replace real data but reduces dependency (e.g., 300 real + 3,000 synthetic ≈ 700 real samples).
   - Excessive synthetic data may introduce noise or reduce generalization (e.g., with 1000 real + 3000 synthetic, the test accuracy is 0.9% lower than using
      only 1000 real)
