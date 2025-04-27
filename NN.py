import tensorflow as tf
from keras.src.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras.layers import *
import matplotlib.pyplot as plt
import scipy
from camera import *

class NN:
    def __init__(self,reset,model_name="my_model"):
        self.model_name = model_name
        self.load_model(reset)

    def load_model(self,reset):
        if reset:
            self.model = keras.models.Sequential()

            self.model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same',
                             data_format='channels_last',
                             input_shape=(28, 28, 1)))
            self.model.add(BatchNormalization())
            self.model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same',
                             data_format='channels_last'))
            self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
            self.model.add(Dropout(0.25))

            self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same',
                             data_format='channels_last'))
            self.model.add(BatchNormalization())
            self.model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                             data_format='channels_last'))
            self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', strides=2))
            self.model.add(Dropout(0.25))

            self.model.add(Flatten())
            self.model.add(Dense(512, activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.25))
            self.model.add(Dense(1024, activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.5))
            self.model.add(Dense(10, activation='softmax'))
            optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
            self.model.compile(optimizer=optimizer,
                               loss ="categorical_crossentropy",
                               metrics=["accuracy"],)

        else:
            self.model = keras.models.load_model(self.model_name)
    def preprocess(self,X,y):
        X = X.reshape(-1,28,28,1)
        X = X.astype("float32")/255
        if type(y)==np.ndarray:
            y = tf.keras.utils.to_categorical(y, num_classes=10)
        return X,y

    def data_augmentation(self,X,y,batch_size):
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.1,  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        datagen.fit(X)
        return datagen.flow(X,y,batch_size=batch_size)

    def learn(self,X_train,y_train,X_test,y_test,epochs=50):
        batch_size = 64
        X_train,y_train = self.preprocess(X_train,y_train)
        X_test,y_test = self.preprocess(X_test,y_test)
        reduce_lr = keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
        checkpoint = keras.callbacks.ModelCheckpoint(self.model_name,
                                                     save_best_only=True,
                                                     save_weights_only=False,
                                                     overwrite=True)
        early_stopping = keras.callbacks.EarlyStopping(
            min_delta=0.001,  # minimium amount of change to count as an improvement
            patience=10,  # how many epochs to wait before stopping
            restore_best_weights=True,)
        history = self.model.fit(self.data_augmentation(X_train, y_train,batch_size), epochs=epochs,
                                      validation_data=(X_test, y_test), verbose=1,
                                      steps_per_epoch=X_train.shape[0] // batch_size,
                                      callbacks=[reduce_lr,checkpoint,early_stopping])  # left out early_stopping parameter as it gets better accuracy


if __name__=="__main__":
    nn = NN(False,"my_second_model")
    camera = Camera()
    X_pred = camera.numbers_images()
    X_pred = nn.preprocess(X_pred,0)[0]
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    print(X_test.shape)
    #nn.learn(X_train,y_train,X_test,y_test,epochs=10)
    X_test,y_test = nn.preprocess(X_test,y_test)
    #print(nn.model.evaluate(X_test,y_test))
    y_pred = nn.model.predict(X_pred)
    str_ = "Les chiffres que tu as écrit sont : "
    for i in range(len(y_pred)-1) :
        str_ += str(np.argmax(y_pred[i]))+", "
    str_+=str(np.argmax(y_pred[len(y_pred)-1]))
    print(str_)

