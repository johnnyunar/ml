from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from keras import datasets, layers, models


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the MNIST dataset and preprocess it"""
    (train_images, train_labels), (test_images, test_labels) = (
        datasets.mnist.load_data()
    )
    train_images = train_images.reshape((-1, 28, 28, 1)) / 255.0
    test_images = test_images.reshape((-1, 28, 28, 1)) / 255.0
    return train_images, train_labels, test_images, test_labels


def build_model() -> models.Model:
    """Define the CNN model"""
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    return model


def train_model(
    model: models.Model,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    epochs: int = 5,
) -> models.Model:
    """Compile and train the CNN model"""
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        train_images,
        train_labels,
        epochs=epochs,
        validation_data=(test_images, test_labels),
    )
    return model


def evaluate_model(
    model: models.Model, test_images: np.ndarray, test_labels: np.ndarray
) -> None:
    """Evaluate the trained model and print accuracy."""
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Accuracy: {test_acc:.4f}")


def plot_predictions(
    model: models.Model, images: np.ndarray, labels: np.ndarray, num_samples: int = 5
) -> None:
    """Plot predictions for a given number of test samples."""
    predictions = model.predict(images[:num_samples])
    for i in range(num_samples):
        plt.imshow(images[i].reshape(28, 28), cmap="gray")
        plt.title(f"Predicted: {np.argmax(predictions[i])}, Actual: {labels[i]}")
        plt.axis("off")
        plt.show()


def main():
    """Run the full training and evaluation pipeline."""
    train_images, train_labels, test_images, test_labels = load_data()
    model = build_model()
    model = train_model(model, train_images, train_labels, test_images, test_labels)
    evaluate_model(model, test_images, test_labels)
    plot_predictions(model, test_images, test_labels)


if __name__ == "__main__":
    main()
