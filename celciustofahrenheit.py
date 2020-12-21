import tensorflow as tf
import numpy as np
celsius = np.array([-40, -10,  0,  8, 15, 22,  38])
fahrenheit = np.array([-40,  14, 32, 46, 59, 72, 100])
for i,c in enumerate(celsius):
print(c, "degrees Celsius = ", fahrenheit[i], "degrees Farenheit")
model = tf.keras.Sequential([
tf.keras.layers.Dense(units = 1, input_shape = [1])
])
model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius, fahrenheit, epochs = 500, verbose = False)
print("Model Training is finished!")
prediction = float(input("Hi, I'm your computer! I predict celsius to farenheit! Give me any number in degrees celcius! :D "))
print(model.predict([prediction]))
import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
