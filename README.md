# Deliverable 1

## Group definition
- Simão Santos 119042
- Gabriel Gonçalves 119550

## Context and motivation of the problem
Static gesture recognition for ASL (American Sign Language), because image classification seems interesting and this is a good baseline problem for it. We did not choose PSL (Portuguese Sign Language) due to the lack of available datasets. Following the suggestion given in class, we changed the project from video recognition to image recognition, and we will focus on a smaller problem with 3 or 4 gesture classes.

## Dataset description (origin, limitations, potential bias)
- We are planning to combine two datasets from Kaggle: [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data), since it is a widely known and used ASL dataset with a large amount of training data, and [Hand Gestures for Human-Robot Interaction](https://www.kaggle.com/datasets/joelbaptista/hand-gestures-for-human-robot-interaction), which includes more difficult conditions related to background and lighting.
- The idea is to use only 3 or 4 gestures that exist in both datasets, so that we can make a direct comparison between simpler and more challenging image conditions.
- One limitation is that the two datasets do not seem to have the same level of complexity, which may create bias during training and evaluation.
- Another limitation is that the number of classes will be reduced from the full alphabet to only a few gestures, so the problem becomes more controlled but also less representative of full ASL recognition.

## Type of ML problem (classification or regression)
This is a classification problem.

## Initial risks and assumptions
- One initial risk is dataset quality, since the images always have the same background, althought this could be mitigated by the use of mediapipe. Also this is solved by the newly added Hand Gestures dataset which contains a large amount of variance in these.
- Another risk is that some letters may be visually very similar, which can make the classification task harder and lead to confusion between classes.
- The model may perform poorly in a real world test basis due to the  bias of the images which are all from the same person in the same background only varying in lighting. (might be fixed by the new dataset)
- Some images may be a little too dark for mediapipe to properly detect the necessary points in the hand. 
- The Hand gestures dataset provides very blurry images whihc might hinder the recognition process.
