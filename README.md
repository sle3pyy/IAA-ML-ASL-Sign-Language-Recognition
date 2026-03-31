# Deliverable 1

## Group definition
- Simão Santos 119042
- Gabriel Gonçalves 119550

## Context and motivation of the problem
ASL (American Sign Language) language recognition, because image classification seems interesting and this is a good baseline problem for it. We did not choose PSL (Portuguese Sign Language) due to the lack of available datasets.

## Dataset description (origin, limitations, potential bias)
- We are planning to use [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data), since it is a widely known and used ASL dataset, with a large amount of training data.  
- One of the reasons for choosing it is the high variance in environments, people, lighting, and backgrounds.

## Type of ML problem (classification or regression)
This is a classification problem.

## Initial risks and assumptions
- One initial risk is dataset quality, since the images may have differences in resolution, hand position, lighting, and background.
- Another risk is that some letters may be visually very similar, which can make the classification task harder and lead to confusion between classes.
- We are also assuming that the dataset labels are mostly correct, even though there may still be some labeling mistakes.
- We also assume that the dataset is varied enough for the model to generalize well, although if the images are too controlled, the model may perform worse on real-world data.
- We will be dependant on mediapipe if that is used. 
