import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from camera import *

class NN:
    def __init__(self,reset,model_name="my_model"):
        self.model_name = model_name
        self.load_model(reset)

    def load_model(self,reset):
        if reset:
            self.model = keras.Sequential([
                keras.layers.Flatten(input_shape=(28, 28)),
                keras.layers.Normalization(),
                keras.layers.Dense(50, activation='relu',kernel_initializer="he_normal"),
                keras.layers.Dense(50, activation='relu', kernel_initializer="he_normal"),
                keras.layers.Dense(10, activation='softmax')
            ])

            optimizer = keras.optimizers.Nadam(learning_rate=0.0001)
            self.model.compile(optimizer=optimizer,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        else:
            self.model = keras.models.load_model(self.model_name)


    def learn(self,X_train,y_train,epochs):
        checkpoint = keras.callbacks.ModelCheckpoint(self.model_name,
                                                     save_best_only=True,
                                                     save_weights_only=False,
                                                     overwrite=True)
        self.model.fit(X_train,y_train,epochs=epochs,validation_split=0.2,callbacks=[checkpoint])


if __name__=="__main__":
    nn = NN(False,"my_first_model")
    camera = Camera()
    X_pred = camera.numbers_images()
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    print(X_test.shape)
    #nn.learn(X_train,y_train,50)
    print(nn.model.evaluate(X_test,y_test))
    y_pred = nn.model.predict(X_pred)
    str_ = "Les chiffres que tu as écrit sont : "
    for i in range(len(y_pred)-1) :
        str_ += str(np.argmax(y_pred[i]))+", "
    str_+=str(np.argmax(y_pred[len(y_pred)-1]))
    print(str_)

