from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential


class LeNet5:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    def build(self):
        model = Sequential()

        model.add(
            Conv2D(
                filters=6,
                kernel_size=(3, 3),
                input_shape=self.input_shape,
                activation="relu",
            )
        )
        model.add(MaxPooling2D())

        model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D())

        model.add(Flatten())
        model.add(Dense(units=120, activation="relu"))
        model.add(Dense(units=84, activation="relu"))
        model.add(Dense(units=1, activation="sigmoid"))

        return model
