# Outputs

This directory contains all the outputs of the [notebook](./../index.ipynb) after running.

For this particular deliverable, the outputs consists of a confusion matrix and its model in a name format following this:

```
(model|confusion_matrix)(\d+) - (\d+\.\d+)%.png
```

Whereas:

- The first capturing group (`$1`) is the type of output;
- The second capturing group (`$2`) is the unix time;
- And the third capturing group (`$3`) is the accuracy of the model.

## Directories

The directories serves as a group on what configuration the model is using, allowing easier identification of the outputs.

**Directory List**:

- [`original`](#original)
- [`thee-conv2d`](#three-conv2d)
- [`no-reduced-lr`](#no-reduced-lr)
- [`no-dropout-and-batchnorm`](#no-dropout-and-batchnorm)
- [`initial`](#initial)

### `original`

The `original` directory is the final configuration, which has the highest rating of them all.

It uses the following layers:

```python
# Input Layers
inputLayers = [
    # Start with 4 filters, 3x3 kernel, ReLU activation. Started with only 4
    # since the image is only 32x32. This allows the model to get low-level
    # features from the image.
    Conv2D(16, (3, 3), activation = 'relu', padding = 'same', input_shape = (32, 32, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Middle with 8 filters, 3x3 kernel, ReLU activation
    # This allows the model to get mid-level features from the image.
    Conv2D(32, (3, 3), activation = 'relu', padding = 'same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # End with 32 filters, 3x3 kernel, ReLU activation
    # This allows the model to get high-level features from the image.
    Conv2D(64, (3, 3), activation = 'relu', padding = 'same'),
    BatchNormalization(),
    MaxPooling2D((2, 2))

	# Test if adding a new layer helps the model learn better.
    Conv2D(128, (3, 3), activation = 'relu', padding = 'same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
]

# Middle Layers (Including Hidden Layers)
middleLayers = [
    # Flatten the input to a 1D vector
    Flatten(),

    # Fully connected layer with 512 neurons and ReLU activation
    # This allows the model to learn complex features from the image.
    Dense(512, activation = 'relu'),

    # Dropout layer with >= 0.5 dropout rate
    # Note: If this layer is uncommented, that means the model underfitted.
    # Dropout(0.5),

    # Dropout layer with < 0.5 dropout rate
    # Note: If this layer is uncommented, that means the model overfitted.
    Dropout(0.4),
]

# Fully connected layer with 10 neurons (one for each class)
# Uses SoftMax since this is a multi-class classification problem
outputLayer = Dense(10, activation = 'softmax')
```

With a configuration of:

```python
# Compile
model.compile(
    optimizer = optimizers.Adam(learning_rate = 1e-4),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy'],
    run_eagerly = True
)

# Train
history = model.fit(
    data['train']['processed']['combined']['generator'],
    steps_per_epoch = SPE,
    epochs = 100,
    validation_data = data['train']['processed']['validation']['generator'],
    validation_steps = VS,
    verbose = 2,
    callbacks = callbacks,
)
```

Whereas:

- `SPE` is the "steps per epoch", calculated as `len(generator) // generator.batch_size` or in numerical - `40000 // 32`
- `VS` is the "validation steps", calculated the same as `SPE`.

And dataset from an `ImageDataGenerator`:

```python
ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)
```

**NOTE:**
For the `original`'s entire code, please refer to the [`index notebook`](./../index.ipynb).

### `three-conv2d`

`three-conv2d` serves as a directory output for the 2nd best configuration. Its only difference from the [`original`](#original) was that the model here uses three `Conv2D` instead of the original`s four.

All its other configurations stayed the same.

### `no-reduced-lr`

`no-reduced-lr` is the 3rd to the last configuration I did prior to [`three-conv2d`](#three-conv2d). It has the same model architecture as the `three-conv2d`, but without the `ReduceLROnPlateau` callback, which actually significantly reduced its accuracy.

### `no-dropout-and-batchnorm`

The `no-dropout-and-batchnorm` uses the [`no-reduced-lr`](#no-reduced-lr) configuration, but without the `Dropout` and `BathNormalization` regularization. This is the one of the configurations I've used prior to the addition of the said normalization to improve its accuracy and output.

### `initial`

`initial` is the last `output` directory. It holds the output for the very first configuration used in this deliverable.

It was the same as [`no-dropout-and-batchnorm`](#no-dropout-and-batchnorm) but this time, the `steps_per_epoch` is now a fixed value, along with the `validation_steps`.
