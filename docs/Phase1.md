# Phase 1: Summary for Deliverable 2

## Data processing methodology

In Phase 1, the adopted strategy was transfer learning with the support of InceptionV3 and MediaPipe.
The processing pipeline was as follows:

1. The original images are stored in `src/Data/collected`.
2. MediaPipe is used to detect hands and extract the valid images.
3. The detected bounding box is used to crop only the hand region.
4. Each valid crop is resized to `299x299`, which is compatible with the InceptionV3 backbone.
5. The processed images are stored in `src/Data/processed`.
6. The processed dataset is then split into:
   - `80%` for the development set (`src/Data/split/train`)
   - `20%` for the test set (`src/Data/split/test`)
7. Validation is derived randomly from test data consisting of `20%` of the total training data.

In addition to dataset preparation, a data augmentation pipeline was introduced and applied only to the training set, including brightness, contrast, saturation, zoom, and small translations. Horizontal flip was not used because in sign language the orientation of the hand may alter the meaning of the gesture.

## Explored models and justification for their selection

The main model explored in this phase was a transfer learning classifier with `InceptionV3`, using ImageNet-pretrained weights and removing the original classification head. On top of the backbone, the layers `GlobalAveragePooling2D`, `Dense(256)`, `Dropout(0.3)`, and a final `Dense(n_classes, activation="softmax")` layer were added.

The choice of transfer learning with InceptionV3 was motivated by four main reasons:

- The available dataset is still relatively small for training a deep CNN from scratch.
- The problem is visual and benefits from representations already learned on large image datasets.
- InceptionV3 naturally operates with `299x299`, which fits well with the cropped-image pipeline.
- The use of a pretrained backbone makes it possible to obtain a strong baseline with less training time and lower risk of instability.

Two training phases were explored:

- Initial training with the backbone frozen, in order to learn only the classifier head.
- Fine-tuning the last `30` layers of InceptionV3 with a lower learning rate, in order to better adapt the features to hand geometry.

During this phase, learning curves were also analysed to assess whether more data was improving generalisation. The results showed that the dataset is, in general, highly learnable, but that smaller subsets appear to be less representative of the full variability, which justifies using the largest possible number of processed samples.

### Insights into our changes 

- Changed the split from 70/20/10 train/test/val to 80/20 train/test and used random 20% from train to validate after observing weird behaviour from the learning curves - A,B,C dataset

![alt text](../src/rel_data/learning_curve_tl.png)

- After the change the problem seem to persist 

![alt text](../src/rel_data/learning_curve_tl_after.png)

- We felt we might have needed to add some more classes for the curve to be evident. Previous models were tested with A,B and C which are very different. Adding Y and F might make it more evident as B is very close to F and A is fairly similar to Y and L fairly similar to C.

![alt text](../src/rel_data/learning_curve_tl_sense.png)

- As shown in this graph the problem was the vast difference between classes, as when there was slight changes within different classes the curve became the expected result. There appears to be no overfitting with large quantities of training data. In smaller quantities of data the dataset does have some overfitting problems probably due to the lack of generalization

- In conclusion, the current three-class setup (A, B, C) may simply be too easy to reveal overfitting or data-scaling effects clearly as they are too different. We are still working on figuring out wether this is a valid concern or something that isnt necessarily a problem.

## Ethical considerations

- The reduced number of classes makes the problem more controlled, but also less representative of real communication scenarios.
- The use of hand images requires attention to privacy and data provenance, even when the focus is only on the gesture itself.
- A model that performs well under controlled conditions may fail in real-world contexts.

## Adjustments required relative to Deliverable 1

- The focus moved from a more initial and exploratory idea to a concrete and reproducible transfer learning pipeline.
- Instead of using raw images directly, we cropped images to fit the InceptionV3's model necesseties.
- An important normalisation issue was identified and corrected: the `Rescaling` layer had to be placed before the backbone in order to respect the input range expected by InceptionV3.
