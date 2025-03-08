import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_and_preprocess_data(url: str) -> tuple[np.ndarray, pd.Series, StandardScaler]:
    """
    Load the Titanic dataset from a URL, preprocess by encoding categorical features,
    and standardize numerical features.

    Args:
        url (str): URL of the dataset.

    Returns:
        tuple: A tuple containing the scaled features array, target series, and the fitted scaler.
    """
    data = pd.read_csv(url)
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
    target = "Survived"
    data = data[features + [target]].dropna()

    # Encode categorical feature 'Sex'
    encoder = LabelEncoder()
    data["Sex"] = encoder.fit_transform(data["Sex"])

    X = data[features]
    y = data[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def build_model(input_dim: int) -> tf.keras.Model:
    """
    Build and compile a Keras Sequential model for binary classification.

    Args:
        input_dim (int): Number of features in the input.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    """
    Load data, train and evaluate the model, and make a prediction.
    """
    data_url = (
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    )
    X_scaled, y, scaler = load_and_preprocess_data(data_url)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    # Define the features used during training
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
    # Create a DataFrame for the new passenger to include valid feature names
    new_passenger = np.array([[3, 1, 22, 1, 0, 7.25]])
    new_passenger_df = pd.DataFrame(new_passenger, columns=features)
    new_passenger_scaled = scaler.transform(new_passenger_df)
    prediction = model.predict(new_passenger_scaled)
    print(f"Survival Probability: {prediction[0][0]:.4f}")


if __name__ == "__main__":
    main()
