# Deliverable 1

## Group definition
- Simão Santos 119042
- Gabriel Gonçalves 119550

## Context and motivation of the problem
ASL (American Sign Language) language recognition, because video classification seems interesting and this is a good baseline problem for it. We did not choose PSL (Portuguese Sign Language) due to the lack of available datasets.

## Dataset description (origin, limitations, potential bias)
- We are planning to use [WLASL](https://github.com/dxli94/WLASL), since it is a widely known ASL dataset based on publicly available YouTube videos that have already been classified. 
- One of the reasons for choosing it is the high variance in environments, people, lighting, and backgrounds.
- We also considered AUTSL, but it seems to show more bias in lighting, people, background, and similar factors.

## Type of ML problem (classification or regression)
This is a classification problem.

## Initial risks and assumptions
- One initial risk is dataset quality, since the videos come from YouTube and may have differences in resolution, length, camera angle, and noise.
- Another risk is class imbalance, because some signs may have many more samples than others, which can affect training and evaluation.
- Since this is a video-based problem, we assume that we will have enough computational resources to preprocess the data and train at least baseline models in a reasonable amount of time.
- We also assume that the variation in people, backgrounds, and lighting will help the model generalize better, although it can also make the classification task harder.
