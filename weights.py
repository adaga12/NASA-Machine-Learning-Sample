import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
import sys




# Function to generate synthetic data with specified parameters and noise

num_parameters = 3
data_points = 168
error = 0.1
def generate_data(num_samples):
    x_data = np.zeros((num_samples, data_points, 1))
    y_data = np.zeros((num_samples, num_parameters))  # Parameters: Amplitude, Frequency, Phase

    for i in range(num_samples):
        amplitude = np.random.uniform(1.0, 5.0)
        frequency = np.random.uniform(1.0, 5.0)
        phase = np.random.uniform(0, 2 * np.pi)
        timedecay = np.random.uniform(0.1, 5.0)
        constant = np.random.uniform(1.0, 5.0)

        # Generate x values with noise
        x_values = np.linspace(0, 5, data_points)
        noise = np.random.uniform(-error, error, size=len(x_values))
        x_data[i, :, 0] = amplitude * np.sin(frequency * x_values + phase) + noise
        y_data[i, :] = [amplitude, frequency, phase]

    return x_data, y_data



np.random.seed(0)
num_samples_train = 1000000
num_samples_val = 1000000
num_samples_test = 50
x_data_train, y_data_train = generate_data(num_samples_train)
x_data_val, y_data_val = generate_data(num_samples_val)
x_data_test, y_data_test = generate_data(num_samples_test)

loss_fn = tf.keras.losses.MeanSquaredError()

learning_rate = 0.001
optimizer = tf.keras.optimizers.SGD(learning_rate)


dataset = {'train': (x_data_train, y_data_train), 'val': (x_data_val, y_data_val), 'test': (x_data_test, y_data_test)}

x_data_train = x_data_train.reshape((num_samples_train, data_points))
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(data_points,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(data_points, activation='linear')
])
model.add(Dense(num_parameters, activation='linear'))

num_epochs = 100


# Weight Adjustment Algorithm (still exploring this)


for epoch in range(num_epochs):
    epoch_losses = []
    for dataset_type, (inputs, targets) in dataset.items():
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = loss_fn(targets, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        epoch_losses.append(loss.numpy())

        print(f"Epoch {epoch+1}/{num_epochs}, Dataset: {dataset_type}, Loss: {loss.numpy()}")

    total_loss = sum(epoch_losses)
    print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {total_loss}")



predictions_test = model(x_data_test)

output_file_path = f'weight_adjustment_sine_function_training_output_{num_epochs}_epochs_{num_samples_train}_labeled_samples_{num_samples_test}_unlabeled_samples_{error}_error.txt'

original_stdout = sys.stdout

# Redirect sys.stdout to the file
def write_to_file():
    with open(output_file_path, 'w') as output_file:
        sys.stdout = output_file
        print("Parameters for Machine Learning Model")
        print()
        print("Number of Epochs:",num_epochs)
        print("Number of Labeled Samples:",num_samples_train)
        print("Number of Unlabeled Samples:",num_samples_test)
        print("Error:",error)
        print()
        for i in range(num_samples_test):
            print("Predicted Parameters for Unlabeled Dataset",i+1,":")
            print(predictions_test[i])
            print("Actual Parameters for Unlabeled Dataset",i+1,":")
            print(y_data_test[i])
            print()
        sys.stdout = original_stdout
write_to_file()
