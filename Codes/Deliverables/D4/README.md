# Deliverable #2

Second assignment; tackles the [**Deep Convoluted Neural Network (DCNN)**](/Codes/Samples/02%20-%20Convoluted%20Neural%20Network/) wherein we focus in image classification.

## Instructions

Download the [CIFAR-10](https://paperswithcode.com/dataset/cifar-10) dataset from any dataset repository. The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Then train a convolutional neural network model. Also, explore and tweak the model by using Regularization techniques. Show the performance of the model using model history line plot and the classification evaluation using confusion matrix.

Write a report containing the Classification Result. Submit the report file named as D2_LastName.pdf and a zip file named as D2_LastName.zip containing your code.

## Dataset

As per instructions, the dataset came from [CIFAR-10](https://paperswithcode.com/dataset/cifar-10) at [Papers With Code](https://paperswithcode.com) website and directly downloaded from the [CS Toronto Edu - CIFAR-10 website](https://www.cs.toronto.edu/~kriz/cifar.html). The author of this dataset, along with the CIFAR-100 are [Alex Krizhevsky](https://www.cs.toronto.edu/~kriz/index.html), Vinod Nair, and Geoffrey Hinton.

CIFAR-10 contains 60k `32x32` color images in 10 classes, with each class containing 6k images. As per the website, there are 50k training images and 10k test ones. The dataset is divided into five training batches and one test batch, each with 10k images. The test batch contains exactly 1k randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5k images from each class.

The classes defined in the dataset are as follows:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

As per the download site:
> The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.

The downloaded version is the python version, which will not be linked directly in this `README.md` file to direct traffic to the said websites.

## Notes

There are 2 notebooks in this deliverable; the [`index.ipynb`](./index.ipynb) and [`index - config modification.ipynb`](./index%20-%20config%20modification.ipynb). The former is the finalized notebook while the latter is for testing modifications and basically, a playground. The latter is also used to backtrack previous configurations to create visual comparisons with the help of its outputs.

## Acknowledgements

> [Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. 2009](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf#page=34)
