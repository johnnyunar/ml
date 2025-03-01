import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Normalization, StringLookup
from typing import Tuple, List


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    print("Loading dataset...")
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully! Here’s a preview:")
    print(data)
    return data


def handle_missing_values(data: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Fill missing values in numeric columns with their median."""
    print("Handling missing values...")
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    print("Missing values handled. Here’s a preview:")
    print(data)
    return data


def encode_categorical_features(data: pd.DataFrame, categorical_col: str) -> tf.Tensor:
    """Encode categorical variables using TensorFlow StringLookup."""
    print("Encoding categorical variables...")
    lookup = StringLookup(output_mode="one_hot", dtype=tf.string)
    lookup.adapt(tf.constant(data[categorical_col].astype(str)))
    categorical_encoded = lookup(tf.constant(data[categorical_col].astype(str).values))
    categorical_encoded = tf.cast(categorical_encoded, tf.float32)
    print("Categorical encoding complete. Example output:")
    print(categorical_encoded.numpy())
    return categorical_encoded


def scale_numeric_features(data: pd.DataFrame, numeric_cols: List[str]) -> tf.Tensor:
    """Apply feature scaling to numeric features using TensorFlow Normalization."""
    print("Applying feature scaling...")
    norm_layer = Normalization()
    numeric_features = tf.constant(data[numeric_cols].values, dtype=tf.float32)
    norm_layer.adapt(numeric_features)
    numeric_features_scaled = norm_layer(numeric_features)
    print("Feature scaling complete. Example output:")
    print(numeric_features_scaled.numpy())
    return numeric_features_scaled


def preprocess_data(file_path: str) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
    """Load, preprocess, and split the dataset into training and testing sets."""
    data = load_data(file_path)
    numeric_cols = ["square_feet", "bedrooms", "bathrooms", "price"]
    categorical_col = "city"

    data = handle_missing_values(data, numeric_cols)
    categorical_encoded = encode_categorical_features(data, categorical_col)
    numeric_features_scaled = scale_numeric_features(
        data, numeric_cols[:-1]
    )  # Exclude price from scaling

    print("Combining processed features...")
    processed_data = tf.concat([numeric_features_scaled, categorical_encoded], axis=1)
    print("Data processing complete. Here’s what the final dataset looks like:")
    print(processed_data.numpy()[:5])

    print("Splitting data into training and testing sets...")
    data_tensor = tf.data.Dataset.from_tensor_slices(processed_data)
    data_list = list(data_tensor.as_numpy_iterator())
    train_size = int(0.8 * len(data_list))
    train_data = data_list[:train_size]
    test_data = data_list[train_size:]
    print(f"Training set size: {len(train_data)} rows")
    print(f"Testing set size: {len(test_data)} rows")

    return train_data, test_data


if __name__ == "__main__":
    # Run preprocessing
    train_data, test_data = preprocess_data("data/101_3_housing.csv")
