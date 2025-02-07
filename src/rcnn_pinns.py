import tensorflow as tf
import numpy as np
import os
import matplotlib.animation as animation
import matplotlib.pyplot as plt


# Define the RCNN model
def rcnn_model(input_shape, layers):
    input_layer = tf.keras.Input(shape=input_shape)
    hidden = input_layer
    for layer in layers:
        hidden = tf.keras.layers.Conv2D(
            filters=layer, kernel_size=(3, 3),bias_initializer='glorot_uniform', padding='same', activation='sigmoid'
        )(hidden)
    output = tf.keras.layers.Conv2D(
        filters=1, kernel_size=(3, 3), activation= 'sigmoid', padding='same'
    )(hidden)
    return tf.keras.Model(inputs=input_layer, outputs=output)

def apply_boundary_conditions(pred_data):
    # Apply boundary conditions to the input data
    pred_data = pred_data.numpy()
    pred_data[:, 0, :, :] = 0
    pred_data[:, :, 0, :] = 1
    pred_data[:, -1, :, :] = 0
    pred_data[:, :, -1, :] = 0
    return tf.convert_to_tensor(pred_data)

# Custom loss function
def custom_loss(model, input_data , delta_t, alpha=pow(10, -2)):
    # Calculate the loss based on the provided formula
    x, y, b = input_data[:, :, :, 0:1], input_data[:, :, :, 1:2], input_data[:, :, :, 2:3]
    with tf.GradientTape(persistent= True) as tape2:
        tape2.watch([x, y])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x, y, b])
            y_pred = model(tf.concat([x, y, b], axis= -1), training=True) # Shape (, 20, 20, 1)
        first_derivative_x = tape1.gradient(y_pred, x) 
        first_derivative_y = tape1.gradient(y_pred, y)
        time_derivative = tape1.gradient(y_pred, b) 
    sec_deri_x = tape2.gradient(first_derivative_x, x) 
    sec_deri_y = tape2.gradient(first_derivative_y, y) 

    del tape1
    del tape2

    loss = tf.concat([tf.square(time_derivative - alpha * (sec_deri_x + sec_deri_y)), 
                  tf.square(apply_boundary_conditions(y_pred) - y_pred)], axis=-1)   
    loss = tf.reduce_sum(loss)
    return loss

def train_rcnn(input_data_init, layers, steps=10, epochs=100, learning_rate=0.001):
    input_shape = input_data_init.shape  # Input has three channels: [data_numbers, x, y, init_condition]
    cnn_model = rcnn_model(input_shape[1:], layers)
    optimizer = tf.optimizers.Adam(learning_rate)
    batch_size = 5

    for epoch in range(epochs):
        batch_numbers = input_shape[0] // batch_size
        if batch_numbers == 0:
            raise ValueError("Insufficient data for batching. Increase input data or reduce batch size.")

        for batch in range(batch_numbers):
            with tf.GradientTape() as tape:
                # Proper slicing for the current batch
                start_index = batch * batch_size
                max_index = min((batch + 1) * batch_size, input_shape[0])
                input_batch = input_data_init[start_index:max_index, :, :, :]

                # Debug batch shape
                print(f"Epoch {epoch + 1}, Batch {batch + 1}, Input Batch Shape: {input_batch.shape}")

                # Calculate loss
                total_loss = custom_loss(cnn_model, input_batch, delta_t=0.01)

            # Compute gradients and update model parameters
            gradients = tape.gradient(total_loss, cnn_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, cnn_model.trainable_variables))

            print(f"Epoch {epoch + 1}/{epochs}, Batch {batch + 1}/{batch_numbers}, Loss: {total_loss.numpy()}")

    return cnn_model


def plot_results(images, save):        
    fig = plt.figure()
    ims = []
    for image in images:
        im = plt.imshow(image, animated=True, cmap='hot')
        # plt.pause(0.1)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=0)
    if save:
        if not os.path.exists('results'):
            os.makedirs('results')
        ani.save('results/rcnn_pinns.gif', writer='pillow', fps=0.3)
    plt.show()


def init_data(x, y, time_frame):
    shape = x.shape
    time_init = np.ones(shape=(shape[0], shape[1]), dtype=np.float32)
    train_data = []
    for i in range(time_frame):
        time = i*0.01*time_init
        train_data.append(np.stack([x, y, time], axis=-1))
    return tf.convert_to_tensor(train_data, dtype=tf.float32)

def main(input_data, layers,train=True, predict_data=1000, steps=10, epochs=100, learning_rate=0.001):
    if train:
        # Call the training function
        rcnn = train_rcnn(input_data,layers, steps, epochs, learning_rate)
        if not os.path.exists('models'):
            os.makedirs('models')
        tf.keras.models.save_model(rcnn, 'models/rcnn_model.h5')
    else:
        # Load the trained model
        rcnn = tf.keras.models.load_model('models/rcnn_model.h5')
        print("Model loaded successfully!")


    y_pred = rcnn.predict(input_data)

    plot_results(y_pred, save= True)



if __name__ == '__main__':
    size = 40
    time_frame = 50
    x = np.linspace(1, 2, size)
    y = np.linspace(1, 2, size)
    x, y = np.meshgrid(x, y)
    layers = [8, 16,32, 16, 8]
    input_data_init = init_data(x, y, time_frame)
    main(input_data_init,layers, train=True, predict_data=30, steps=30, epochs=100, learning_rate=0.01)
    