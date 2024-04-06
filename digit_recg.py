import tensorflow as tf
import numpy as np
import cv2

mnist = tf.keras.datasets.mnist

(test_images, test_labels), (train_images, train_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(10)
])

model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy']
)

model.fit(train_images, train_labels, epochs = 10)

test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test Accuray: {test_accuracy*100}%")

prediction = model.predict(test_images)
print("Predictions:", prediction)
print(np.argmax(prediction[3]))

cv2.imshow("Image", test_images[3])
cv2.waitKey(0)

if KeyboardInterrupt:
    cv2.destroyAllWindows()
    print("Closed!")
