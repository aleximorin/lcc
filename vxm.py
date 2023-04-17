import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import voxelmorph as vxm
from tensorflow.keras.datasets import mnist
import matplotlib
matplotlib.use("TkAgg")


def generator(x_data, batch_size=32):
    shape = x_data.shape[1:]
    ndims = len(shape)

    zero_phi = np.zeros([batch_size, *shape, ndims])

    while True:

        index1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[index1, ..., np.newaxis]

        index2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[index2, ..., np.newaxis]

        inputs = [moving_images, fixed_images]
        outputs = [fixed_images, zero_phi]

        yield (inputs, outputs)


(x_train, y_train), (x_test, y_test) = mnist.load_data()

pad_amount = ((0, 0), (2,2), (2,2))

# fix data
x_train = np.pad(x_train, pad_amount, 'constant')
x_test = np.pad(x_test, pad_amount, 'constant')

digit = 5
digit_train = x_train[y_train == digit] / 255
digit_test = x_test[y_test == digit] / 255

h, w = digit_test.shape[1:]
ndim = 2
unet_input_features = 2

nb_features = [(h, w, h, w), (h, w, h, w, h, int(w/2))]
inshape = x_train.shape[1:]

vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)

losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

lambda_param = 0.05
loss_weights = [1, lambda_param]

vxm_model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights)

train_generator = generator(digit_train)
in_sample, out_sample = next(train_generator)


nb_epochs = 20
steps_per_epoch = 100
hist = vxm_model.fit(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2)

val_generator = generator(digit_test, batch_size=1)
val_input, _ = next(val_generator)

pred = vxm_model.predict(val_input)

images = [img[0, :, :, 0] for img in val_input + pred]
fig, axs = plt.subplots(1, 4, sharex='all', sharey='all')
for i, ax in enumerate(axs.flatten()):
    ax.imshow(images[i])

print()