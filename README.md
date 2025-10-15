# my-first-project
My first test project




class BostonHousingModel:
    def __init__(self):
        self.model = None
        self.mean = None
        self.std = None

    def load_and_preprocess_data(self):
        (train_data, train_target), (test_data, test_target) = (
            tf.keras.datasets.boston_housing.load_data()
        )

        self.mean = train_data.mean(axis=0)
        self.std = train_data.std(axis=0)
        train_data = (train_data - self.mean) / self.std
        test_data = (test_data - self.mean) / self.std

        return train_data, train_target, test_data, test_target

    def build_model(self):

        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, verbose=1, restore_best_weights=True
        )

        inputs = tf.keras.Input(shape=(13,))

        x_layers = tf.keras.layers.Dense(128, activation="relu")(inputs)
        x_layers = tf.keras.layers.BatchNormalization()(x_layers)
        x_layers = tf.keras.layers.Dense(64, activation="relu")(x_layers)
        x_layers = tf.keras.layers.Dense(32,activation="relu")(x_layers)
        x_layers = tf.keras.layers.Dropout(0.2)(x_layers)


        outputs = tf.keras.layers.Dense(1)(x_layers)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        return callback

    def fit_model(self, callback, train_data, train_target, test_data, test_target):

        history = self.model.fit(
            train_data,
            train_target,
            batch_size=32,
            epochs=100,
            validation_data=(test_data, test_target),
            callbacks=[callback],
            verbose=1,
        )

        return history

    def evaluate(self, test_data, test_target):

        loss, mae = self.model.evaluate(test_data, test_target, verbose=0)
        print(f"测试集准确率：{mae:.3f}")
        print(f"测试集损失率：{loss:.3f}%")

        return loss, mae

    def predict_house_price(self, sample):
        if self.model is None:
            raise ValueError("模型尚未创建,请先调用build_model()")

        sample = (sample - self.mean) / self.std
        sample = np.expand_dims(sample,axis=0)
        pred = self.model.predict(sample,verbose=0)
        print(f"预测房价：{pred[0][0]:.2f}万美元,实际价格是:{test_target[7]:.2f}万美元")

        return pred[0][0]


if __name__ == "__main__":

    import tensorflow as tf
    import numpy as np

    boston = BostonHousingModel()

    train_data, train_target, test_data, test_target = boston.load_and_preprocess_data()

    callback = boston.build_model()

    history = boston.fit_model(
        callback, train_data, train_target, test_data, test_target
    )

    boston.evaluate(test_data, test_target)

    boston.predict_house_price(test_data[7])
