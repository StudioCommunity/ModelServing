import keras
import logging
import os
import json
import yaml
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

from builtin_models.keras import save_model

# Test dynamic install package
from pip._internal import main as pipmain
pipmain(["install", "click"])
import click

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.info(f"in dstest echo")
logger = logging.getLogger(__name__)


@click.command()
@click.option('--model_path', default="./model/")
def run_pipeline(
    model_path
    ):
    batch_size = 64
    num_classes = 10
    epochs = 5
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data() # x_train shape (60000, 28, 28), y_train shape (60000, 1). x_test(10000, 28, 28)

    x_train = x_train.reshape(x_train.shape[0], -1) # equals x_train.reshape(x_test.shape[0], 784). 28*28=784
    x_test = x_test.reshape(x_test.shape[0], 784) # 28*28=784

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255 #value change to [0,1]
    x_test /= 255

    # convert class vectors to binary class matrices, e.g. y_train[0] = 5, img is 5,after to_categorical, y_train[0] is  [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Dense(512,activation='relu',input_shape=(784,)))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    #model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

    save_model(model, model_path, conda_env=None)

    logger.info(f"training finished")

# python -m dstest.keras.trainer_and_save_model_h5  --model_path model/keras-mnist
if __name__ == '__main__':
    run_pipeline()
    



