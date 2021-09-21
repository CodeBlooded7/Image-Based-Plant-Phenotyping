import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history["val_loss"])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()


class myCall(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.97):
            self.model.stop_training = True


data_x = np.load("train_x.npy")
data_y = np.load("train_y.npy")
val_x = np.load("val_x.npy")
val_y = np.load("val_y.npy")

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(16, (3, 3), input_shape=(256, 256, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(3, 3))
# model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
# model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Dense(3, activation='softmax'))

# model.load_weights("model.h5")

model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
history = model.fit(data_x, data_y, validation_data=(val_x, val_y), shuffle=True, verbose=1, epochs=20, batch_size=32, callbacks=[myCall()])

model.summary()

plot_graphs(history, "loss")

ti = input("save weights?(Y/N)")
if ti == "Y":
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
