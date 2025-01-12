import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os 
import tensorflow as tf

def my_net(shape, layers):
    input_layer = tf.keras.layers.Input(shape)
    x = input_layer
    output_layer = []
    for i in layers:
        x = tf.keras.layers.Dense(units= i, activation= 'relu')(x)
        output_layer.append(x)
    x = tf.keras.layers.Dense(units= 1)(x)
    output_layer.append(x)
    return tf.keras.Model(input_layer, output_layer)

def train(x, y, model, epochs: int):
    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
    for i in range(epochs):
        with tf.GradientTape() as tape:
            pred = model(x)[-1]
            loss = tf.reduce_mean(tf.square(pred - y))  
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        if i % 50 == 0:
            print(f"Epoch {i}, Loss: {loss.numpy()}")
    return model

if __name__ == '__main__':

    x = np.linspace(-10, 10, 100).reshape(-1, 1)  # Reshape for model input
    y = np.sin(x).reshape(-1, 1)  # Reshape for compatibility

    # Model setup
    alpha = 0.01
    layers = [7, 7, 7, 7, 7]
    model = my_net((1,), layers=layers)
    trainable = 1

    if trainable:
        result = train(x, y, model= model, epochs= 2000)
        if not os.path.exists('training_result'):
            os.mkdir('training_result')
        result.save('./training_result/model_morelayer.h5')
        result = result.predict(x)
    else:
        model = tf.keras.models.load_model('./training_result/model_morelayer.h5')
        result = model.predict(x)
    layer1 = np.sum(result[0], axis= 1)
    layer2 = np.sum(result[1], axis= 1)
    layer3 = np.sum(result[2], axis= 1)
    layer4 = np.sum(result[3], axis= 1)
    layer5 = np.sum(result[4], axis= 1)
    layer6 = result[-1]

    # Plot the output and hidden layers
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label="True Data", color="blue")
    plt.plot(x, layer1, label="layer1", color="green", linestyle="dashed")
    plt.plot(x, layer2, label="layer2", color="green", linestyle="dashed")
    plt.plot(x, layer3, label="layer3", color="yellow", linestyle="dashed")
    plt.plot(x, layer4, label="layer4", color="yellow", linestyle="dashed")
    plt.plot(x, layer5, label="layer5", color="red", linestyle="dashed")
    plt.plot(x, layer6, label="Model Prediction", color="red", linestyle="dashed")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Sine Wave Fit with Neural Network")
    if not os.path.exists('results'):
        os.mkdir('results')
    plt.savefig('./results/hidden_layers.png')
    plt.show()

