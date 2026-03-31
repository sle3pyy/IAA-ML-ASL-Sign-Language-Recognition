# Deliverable 1

## Group definition
- Simão Santos 119042
- Gabriel Gonçalves 119550

## Context and motivation of the problem
ASL (American Sign Language) language recognition, because image classification seems interesting and this is a good baseline problem for it. We did not choose PSL (Portuguese Sign Language) due to the lack of available datasets.

## Dataset description (origin, limitations, potential bias)
- We are planning to use [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data), since it is a widely known and used ASL dataset, with a large amount of training data.
- The images are all from the same person in the same background only varying in lighting which may worsen the models accuracy in different environments. 

## Type of ML problem (classification or regression)
This is a classification problem.

## Initial risks and assumptions
- One initial risk is dataset quality, since the images always have the same background, althought this could be mitigated by the use of mediapipe.
- Another risk is that some letters may be visually very similar, which can make the classification task harder and lead to confusion between classes.
- The model may perform poorly in a real world test basis due to the  bias of the images which are all from the same person in the same background only varying in lighting.
- Some images may be a little too dark for mediapipe to properly detect the necessary points in the hand. 
