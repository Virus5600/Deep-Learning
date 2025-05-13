from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications import VGG16, InceptionV3 # type: ignore
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, Resizing # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.regularizers import l1, l2, l1_l2 # type: ignore
from tensorflow.keras.utils import plot_model # type: ignore
from typing import Union
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import seaborn as sns
import tensorflow as tf

def test() -> None:
	"""
	A test function to check if the module is working.
	"""
	print("Test function called.")
	return None

def unpickle(file):
	"""
	Decompiles a pickle file.

	:param file: Path to the pickle file.
	:type file: str
	
	:return: The unpickled data.
	:rtype: dict
	"""
	if not os.path.isfile(file):
		raise FileNotFoundError(f"File {file} not found.")
	
	with open(file, 'rb') as fo:
		data = pickle.load(fo, encoding='bytes')
	return data

def unpickleToTuple(file) -> tuple:
	"""
	Decompiles a pickle file into a tuple. The tuple
	contains the `data` and `labels` keys from the unpickled data
	in that respective order.

	The `data` key contains the image data and the `labels` key
	contains the labels for the images.

	The `data` key is a numpy array of shape (n, 3072) where n is the
	number of images. The `labels` key is a list of length n containing
	the labels for the images.

	:param file: Path to the pickle file.
	:type file: str
	
	:return: The unpickled data as a tuple.
	:rtype: tuple
	"""
	data = unpickle(file)
	return (data[b'data'], data[b'labels'])

def showImg(input, title = None, axis = False) -> None:
	"""
	Displays an image.

	:param input: The image to display.
	:type input: numpy.ndarray
	
	:param title: Title of the image.
	:type title: str

	:param axis: Whether to show the axis or not.
	:type axis: bool
	"""
	plt.imshow(input)
	if title is not None:
		plt.title(title)
	if not axis:
		plt.axis('off')
	plt.show()

	return None

def buildModel(baseModel, trainBase = False, poolingLayer = GlobalAveragePooling2D, dropoutRate = 0.5, denseUnits = 512, useBatchNorm = True, hlDecayMode = 'l2', hlDecayRate = 0.001, opDecayMode = 'l2', opDecayRate = 0.001) -> Model:
	"""
	Builds the model for transfer learning.

	:param baseModel: The base model to use for transfer learning. Required.
	:type baseModel: tensorflow.keras.applications.VGG16 | tensorflow.keras.applications.InceptionV3 | tensorflow.keras.applications.ResNet50

	:param trainBase: Whether to train the base model or not. Default is False.
	:type trainBase: bool

	:param dropoutRate: The dropout rate for the model. Default is 0.5.
	:type dropoutRate: float

	:param denseUnits: The number of units in the dense layer. Default is 512.
	:type denseUnits: int

	:param useBatchNorm: Whether to use batch normalization or not. Default is True.
	:type useBatchNorm: bool

	:hlDecayMode: The decay mode for the hidden layer. Default is 'l2'.
	:type hlDecayMode: 'l1', 'l2', 'l1_l2', 'none'

	:hlDecayRate: The decay rate for the hidden layer. Default is 0.001.
	:type hlDecayRate: float

	:opDecayMode: The decay mode for the output layer. Default is 'l2'.
	:type opDecayMode: 'l1', 'l2', 'l1_l2', 'none'

	:opDecayRate: The decay rate for the output layer. Default is 0.001.
	:type opDecayRate: float

	:return: The built model.
	:rtype: tensorflow.keras.models.Model
	"""
	# Guard check just to make sure the pooling layer won't kill my device...
	# Basically, it ensures that 'Flatten' is not used with models other than VGG16
	if poolingLayer is Flatten and baseModel is not VGG16:
		poolingLayer = GlobalAveragePooling2D

	# Applies the resizing layer to the input shape
	targetShape = (299, 299) if baseModel is InceptionV3 else (224, 224)
	inputTensor = Input(shape = (32, 32, 3))
	inputTensor = Resizing(*targetShape)(inputTensor)
	print("Resizing Layer Shape: ", inputTensor.shape)

	base = baseModel(weights = 'imagenet', include_top = False, input_tensor = inputTensor)
	print(f"Base model: {baseModel.__name__} - {base.output}")

	# Freeze the base model layers (if True)
	base.trainable = trainBase

	x = poolingLayer()(base.output)
	print(f"Pooling Layer Shape: {x.shape}")

	if useBatchNorm:
		x = BatchNormalization()(x)

	# HIDDEN LAYER
	# By default, this looks like without the decays:
	#	x = Dense(512, activation = "relu")(x)
	hlKernelRegularizer = None
	if hlDecayMode in ('l1', 'l2', 'l1_l2') and hlDecayRate > 0:
		if hlDecayMode == 'l1':
			hlKernelRegularizer = l1(hlDecayRate)
		elif hlDecayMode == 'l2':
			hlKernelRegularizer = l2(hlDecayRate)
		elif hlDecayMode == 'l1_l2':
			if hlDecayRate is list:
				hlKernelRegularizer = l1_l2(hlDecayRate[0], hlDecayRate[1])
			else:
				hlKernelRegularizer = l1_l2(hlDecayRate, hlDecayRate)
	x = Dense(denseUnits, activation = "relu", kernel_regularizer = hlKernelRegularizer)(x)

	if useBatchNorm:
		x = BatchNormalization()(x)

	if dropoutRate > 0:
		# By default, this looks like:
		#	x = Dropout(0.5)(x)
		x = Dropout(dropoutRate)(x)

	# OUTPUT LAYER
	# By default, this looks like without the decays:
	#	x = Dense(10, activation = "softmax")(x)
	opKernelRegularizer = None
	if opDecayMode in ('l1', 'l2', 'l1_l2') and opDecayRate > 0:
		if opDecayMode == 'l1':
			opKernelRegularizer = l1(opDecayRate)
		elif opDecayMode == 'l2':
			opKernelRegularizer = l2(opDecayRate)
		elif opDecayMode == 'l1_l2':
			if opDecayRate is list:
				opKernelRegularizer = l1_l2(opDecayRate[0], opDecayRate[1])
			else:
				opKernelRegularizer = l1_l2(opDecayRate, opDecayRate)
	outputs = Dense(10, activation = "relu", kernel_regularizer = opKernelRegularizer)(x)

	return Model(inputs = inputTensor, outputs = outputs)

def randomConfigSampler(possibleConfigValues, nSamples) -> list:
	"""
	Samples random configurations from the possible configuration values. The function is
	designed to generate a list of unique configurations, allowing for creating a diverse set of
	models for experimentation.

	:param popssibleConfigValues: The possible configuration values.
	:type popssibleConfigValues: dict

	:param nSamples: The number of samples to generate.
	:type nSamples: int

	:return: A list of random configurations.
	:rtype: list
	"""
	configs = []
	definedConfigs = set()

	while len(configs) < nSamples:
		config = {}
 
		# Sample a random configuration
		for key, value in possibleConfigValues.items():
			config[key] = random.choice(value)

			if key == 'poolingLayer':
				# Additional guard check just to make sure the pooling layer won't kill my device...
				# Basically, it ensures that 'Flatten' is not used with models other than VGG16
				if config[key] is Flatten and config['baseModel'] is not VGG16:
					config[key] = GlobalAveragePooling2D

		# Creates the input shape based on the base model
		# target_shape = (299, 299) if config['baseModel'] is InceptionV3 else (224, 224)
		# config['inputTensor'] = Input(shape = (*target_shape, 3))

		configTuple = tuple(sorted((k, v) for k, v in config.items() if k != 'inputTensor'))

		# Add the config to the list if unique
		if configTuple not in definedConfigs:
			definedConfigs.add(configTuple)
			configs.append(config)
		# If the config is not unique, try again
		else:
			continue

	return configs

def preprocessImg(img, lbl, targetSize, rescale = True, augment = False) -> tuple:
	"""
	Preprocesses the image and label for training.

	:param img: The image to preprocess.
	:type img: numpy.ndarray

	:param lbl: The label to preprocess.
	:type lbl: int

	:param targetSize: The target size for the image.
	:type targetSize: tuple

	:param rescale: Whether to rescale the image or not. Default is True.
	:type rescale: bool

	:param augment: Whether to augment the image or not. Default is False.
	:type augment: bool

	:return: The preprocessed image and label.
	:rtype: tuple
	"""
	img = tf.reshape(img, (32, 32, 3))
	img = tf.image.resize(img, targetSize)

	if rescale:
		img = tf.cast(img, tf.float32) / 255.0

	if augment:
		img = tf.image.random_flip_left_right(img)
		img = tf.image.random_brightness(img, 0.2)
		img = tf.image.random_contrast(img, 0.8, 1.2)
		img = tf.image.random_saturation(img, 0.8, 1.2)
		img = tf.image.random_hue(img, 0.08)

	return img, lbl

def makeDataset(x, y, targetSize, rescale = True, augment = False, batchSize = 15) -> tf.data.Dataset:
	"""
	Creates a TensorFlow dataset from the given data.

	:param x: The input data.
	:type x: numpy.ndarray

	:param y: The labels.
	:type y: numpy.ndarray

	:param targetSize: The target size for the images.
	:type targetSize: tuple

	:param rescale: Whether to rescale the images or not. Default is True.
	:type rescale: bool

	:param augment: Whether to augment the images or not. Default is False.
	:type augment: bool

	:param batchSize: The batch size for the dataset. Default is 15.
	:type batchSize: int
	"""
	ds = tf.data.Dataset.from_tensor_slices((x, y))
	ds = ds.map(lambda img, label: preprocessImg(img, label, targetSize, rescale, augment),
			num_parallel_calls=tf.data.AUTOTUNE)
	if augment:
		ds = ds.shuffle(10000)
	ds = ds.batch(batchSize).prefetch(tf.data.AUTOTUNE)

	return ds

def getMetrics(model, dataset, logPerBatch = False) -> tuple:
	"""
	Calculates the accuracy `(avg)` and accuracy range `(min, max)`
	for the given model and dataset.

	Also returns the true labels and predicted labels. The metrics are all floats
	that represent the score in decimal and not percentage.

	:param model: The model to use for prediction.
	:type model: tensorflow.keras.models.Model

	:param dataset: The dataset to calculate the metrics for.
	:type dataset: tensorflow.data.Dataset

	:param logPerBatch: Whether to log the metrics per batch or not. Default is False.
	:type logPerBatch: bool

	:return: A tuple containing metrics, the true labels, and the predicted labels; wherein the metrics is also a tuple containing the `(avg, min, max)` values.
	:rtype: tuple(tuple, list, list)
	"""
	min = 0
	max = 0
	avg = 0

	yTrue = []
	yPred = []

	for x, y in dataset:
		classes = model.predict(x, verbose = 0)
		classes = np.argmax(classes, axis = 1)

		yTrue.extend(y)
		yPred.extend(classes)

		if logPerBatch:
			print(f"Classes: {classes}")
			print(f"Labels: {y}")

		# Accuracy
		accuracy = np.sum(classes == y) / len(y)
		if accuracy > max:
			max = accuracy
		if accuracy < min or min == 0:
			min = accuracy
		avg += accuracy

		if logPerBatch:
			print(f"Formula: {np.sum(classes == y)} / {len(y)}")
			print(f"Accuracy: {accuracy * 100:.2f}%")
			print(f"Of {len(y)} images, {np.sum(classes == y)} were correct while {np.sum(classes != y)} were incorrect.")

	avg /= len(dataset)
	return (avg, min, max), yTrue, yPred

def plotModelHistory(modelName, history, accuracy, yTrue, yPred) -> None:
	"""
	Plots the training history of the model.

	:param modelName: The name of the model.
	:type modelName: str

	:param history: The training history of the model.
	:type history: tensorflow.keras.callbacks.History

	:param accuracy: The accuracy of the model in decimal form (not percentage).
	:type accuracy: float

	:param yTrue: The true labels of the dataset.
	:type yTrue: list

	:param yPred: The predicted labels of the dataset.
	:type yPred: list
	"""
	status = "Underfitted" if accuracy < 0.5 else "Overfitted" if accuracy > 0.9 else "Just Right"
	yTrue = np.sum(np.array(yTrue) == np.array(yPred))
	yLength = len(yPred)
	yScore = yTrue / yLength

	accuracy = accuracy * 100
	unixTime = int(datetime.datetime.now().timestamp() * 1e6)

	print(f"Accuracy: {accuracy:.2f}%")
	print(f"Using `forCM`: {yTrue} / {yLength} = {yScore * 100:.2f}%")

	if not os.path.exists(f"outputs/accuracy/{modelName}"):
		os.makedirs(f"outputs/accuracy/{modelName}")

	plt.figure(figsize = (10, 6))
	plt.plot(history.history['accuracy'], color = 'blue', label = 'train')
	plt.plot(history.history['val_accuracy'], color = 'red', label = 'val')
	plt.legend()
	plt.grid()
	plt.title(f'Accuracy ({modelName})\nStatus: {status} ({accuracy:.2f}%)')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.savefig(f"outputs/accuracy/{modelName}/{unixTime} - {accuracy:.2f}%.png")

def plotConfusionMatrix(yTrue, yPred, modelName, accuracy) -> None:
	"""
	Plots the confusion matrix for the model predictions.

	:param yTrue: The true labels of the dataset.
	:type yTrue: list

	:param yPred: The predicted labels of the dataset.
	:type yPred: list

	:param modelName: The name of the model.
	:type modelName: str

	:param accuracy: The accuracy of the model in decimal form (not percentage).
	:type accuracy: float
	"""
	unixTime = int(datetime.datetime.now().timestamp() * 1e6)
	confusion = confusion_matrix(yTrue, yPred)

	plt.figure(figsize = (10, 8))
	sns.heatmap(confusion, annot = True, fmt = 'd', cmap = 'Blues')

	plt.title('Confusion Matrix')
	plt.xlabel('Predicted Label')
	plt.ylabel('True Label')

	if not os.path.exists(f"outputs/confusion_matrix/{modelName}"):
		os.makedirs(f"outputs/confusion_matrix/{modelName}")

	plt.savefig(f"outputs/confusion_matrix/{modelName}/{unixTime} - {accuracy:.2f}%.png")

def plotModel(model, accuracy, modelName = None) -> None:
	"""
	Plots the model architecture.

	:param model: The model to plot.
	:type model: tensorflow.keras.models.Model

	:param accuracy: The accuracy of the model in decimal form (not percentage).
	:type accuracy: float

	:param modelName: The name of the model. Optional.
	:type modelName: str
	"""
	if modelName is None:
		modelName = model.__name__

	unixTime = int(datetime.datetime.now().timestamp() * 1e6)
	plot_model(model, to_file = f"outputs/{modelName}_model_{unixTime} - {accuracy}%.png", show_shapes = True, show_layer_names = True)

def getModelLayerNames(model: Model) -> list:
	"""
	Gets the layer names of the model.

	:param model: The model to get the layer names from.
	:type model: tensorflow.keras.models.Model

	:return: The layer names of the model.
	:rtype: list
	"""
	return [layer.name for layer in model.layers]

None