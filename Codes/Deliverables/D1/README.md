# Deliverable #1

First assignment that tackles the [Perceptron and Neural Networks](/Codes/Samples/01%20-%20Neural%20Network/).

## Instructions

- Download a multi-classification dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/) or any dataset repository. Then create a neural network model with two (2) hidden layers with ten (10) units each.

- Also, explore by using Regularization techniques. Show the performance of the model using confusion matrix.

## Dataset

The dataset came from [UCI ML Repository](https://archive.ics.uci.edu/) - An [Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) that contains 43 sets; 30 for training and 13 for testing.

Under the "Additional Information" section of the page:
> We used preprocessing programs made available by NIST to extract normalized bitmaps of handwritten digits from a preprinted form. From a total of 43 people, 30 contributed to the training set and different 13 to the test set. 32x32 bitmaps are divided into nonoverlapping blocks of 4x4 and the number of on pixels are counted in each block. This generates an input matrix of 8x8 where each element is an integer in the range 0..16. This reduces dimensionality and gives invariance to small distortions.

For people who isn't that great with technicalities, it basically says that:

The dataset consists of handwritten digits (0-9) collected from 43 individuals.

- **Training Data:** 30 people's handwriting.
- **Test Data:** 13 different people's handwriting.

Each digit is originally a **32x32 pixel** image. To reduce complexity:  

1. The image is divided into **4x4 blocks**, creating an **8x8 grid**.
2. Each block counts the number of "on" pixels (dark areas), giving values from **0 to 16**.

This preprocessing is done to simplify the data while preserving essential patterns, helping the model recognize digits more effectively.
