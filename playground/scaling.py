import pandas as pd
from sklearn.preprocessing import StandardScaler

# Create a simple dataset with features on different scales
data = pd.DataFrame(
    {"Age": [20, 30, 40, 50, 60], "Income": [30000, 50000, 70000, 90000, 110000]}
)
print("Original Data:")
print(data)

# Standardize the numerical features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

print("\nScaled Data:")
print(scaled_df)
