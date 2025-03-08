import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def load_salary_data(file_path: str) -> pd.DataFrame:
    """
    Load the salary dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)


def preprocess_data(
    data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Preprocess the salary data by selecting features, dropping missing values, and standardizing.

    Args:
        data (pd.DataFrame): The raw salary data.

    Returns:
        tuple: Scaled features, target values, and the fitted scaler.
    """
    # Example features and target
    features = ["YearsExperience", "EducationLevel"]
    target = "Salary"

    # Drop rows with missing values
    data = data[features + [target]].dropna()

    X = data[features]
    y = data[target]

    # Standardize features so they have mean 0 and std 1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.to_numpy(), scaler


def train_model(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    """
    Train a linear regression model on the preprocessed data.

    Args:
        X (np.ndarray): Scaled features.
        y (np.ndarray): Target salary values.

    Returns:
        LinearRegression: Trained model.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def predict_salary(
    model: LinearRegression, scaler: StandardScaler, new_data: np.ndarray
) -> float:
    """
    Predict salary for new input data.

    Args:
        model (LinearRegression): Trained regression model.
        scaler (StandardScaler): Fitted scaler used for standardizing features.
        new_data (np.ndarray): New data with raw feature values.

    Returns:
        float: Predicted salary.
    """
    # Scale new data using the same scaler
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    return prediction[0]


def main() -> None:
    """
    Full process: load data, preprocess, train, evaluate, and predict salary.
    """
    # Load the dataset (assume 'salary_data.csv' is in your working directory)
    data = load_salary_data("data/salaries.csv")

    # Preprocess the data
    X_scaled, y, scaler = preprocess_data(data)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    print(f"Test R2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"Test MSE: {mean_squared_error(y_test, y_pred):.4f}")

    # Predict salary for a new person with 5 years of experience and education level 2
    new_input = np.array([[5, 2]])
    predicted_salary = predict_salary(model, scaler, new_input)
    print(f"Predicted Salary: ${predicted_salary:.2f}")


if __name__ == "__main__":
    main()
