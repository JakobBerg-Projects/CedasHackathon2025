
---

### **Here's the Corrected and Reordered Code with Explanations:**

```python
# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, RepeatVector

# %%
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# %%
# Load data
train_data_full = pd.read_parquet("/home/henrik/projects/cedas2025/src/data/cedas2025_material/data/chargecurves_train.parquet")

# %%
# Reshape the dataframe
def reshape_dataframe(df):
    first_timestamps = df.groupby('id')['timestamp'].first().reset_index()
    pivot_df = df.pivot(index=['id', 'nominal_power', 'location_id'],
                        columns='sub_id',
                        values=['soc', 'power']).reset_index()
    pivot_df.columns = [
        f'{col[0]}_{col[1]}' if col[1] != '' else col[0]
        for col in pivot_df.columns
    ]
    result_df = pivot_df.merge(first_timestamps, on='id')
    return result_df

train_data_full = reshape_dataframe(train_data_full)

# %%
# Split the data
# Note: We split the data before setting NaNs to ensure we have the true values in y for validation and test sets.
train, temp = train_test_split(train_data_full, train_size=0.70, test_size=0.30, shuffle=True, random_state=42)
validation, test = train_test_split(temp, train_size=0.5, test_size=0.5, shuffle=True, random_state=42)

# %%
# Function to set future soc and power columns to NaN
def set_columns_to_nan(df, start_range=10, end_range=39):
    soc_cols = [f'soc_{i}' for i in range(start_range, end_range + 1)]
    power_cols = [f'power_{i}' for i in range(start_range, end_range + 1)]
    df.loc[:, soc_cols] = np.nan
    df.loc[:, power_cols] = np.nan
    return df

# %%
# Create copies of validation and test sets before setting NaNs, to preserve true y values
validation_true = validation.copy()
test_true = test.copy()

# %%
# Set future values to NaN in validation and test sets
validation = set_columns_to_nan(validation)
test = set_columns_to_nan(test)

# %%
# Now, create sequences
def create_sequences(df, is_training=True):
    X, y = [], []
    for index, row in df.iterrows():
        # Input sequence: steps 0 to 9
        input_seq = []
        for i in range(10):
            soc = row[f'soc_{i}']
            power = row[f'power_{i}']
            input_seq.append([soc, power])

        X.append(input_seq)

        if is_training:
            # Output sequence: steps 10 to 39
            output_seq = []
            for i in range(10, 40):
                soc = row[f'soc_{i}']
                power = row[f'power_{i}']
                output_seq.append([soc, power])
            y.append(output_seq)
        else:
            # For validation and test sets, use the true values from validation_true and test_true
            output_seq = []
            for i in range(10, 40):
                soc = df.loc[index, f'soc_{i}']  # This will be NaN
                power = df.loc[index, f'power_{i}']  # This will be NaN
                # Get true values from validation_true or test_true
                if pd.isna(soc) or pd.isna(power):
                    soc = validation_true.loc[index, f'soc_{i}'] if df is validation else test_true.loc[index, f'soc_{i}']
                    power = validation_true.loc[index, f'power_{i}'] if df is validation else test_true.loc[index, f'power_{i}']
                output_seq.append([soc, power])
            y.append(output_seq)
    return np.array(X), np.array(y)

# %%
# Create sequences for training, validation, and test sets
X_train, y_train = create_sequences(train, is_training=True)
X_validation, y_validation = create_sequences(validation, is_training=False)
X_test, y_test = create_sequences(test, is_training=False)

# %%
# Check shapes
print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
print(f'X_validation shape: {X_validation.shape}, y_validation shape: {y_validation.shape}')
print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')

# %%
# Build the Model
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, RepeatVector

# Define the model
model = Sequential()
# Encoder
model.add(LSTM(50, activation='relu', input_shape=(10, 2)))
# Repeat the context vector for the length of the output sequence
model.add(RepeatVector(30))
# Decoder
model.add(LSTM(50, activation='relu', return_sequences=True))
# TimeDistributed layer to apply Dense layer to each time step
model.add(TimeDistributed(Dense(2)))
model.compile(optimizer='adam', loss='mse')
# Print the model summary
model.summary()

# %%
# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_validation, y_validation),
    verbose=1
)

# %%
# Make predictions
y_pred = model.predict(X_test)

# %%
# Evaluate the model
from sklearn.metrics import mean_squared_error

# Reshape y_test and y_pred to 2D arrays for evaluation
y_test_flat = y_test.reshape(-1, 2)
y_pred_flat = y_pred.reshape(-1, 2)

mse = mean_squared_error(y_test_flat, y_pred_flat)
print(f'Test MSE: {mse}')
```

---

### **Detailed Explanation of Adjustments:**

#### **1. Data Splitting Order:**

- We **split the data first** before setting future values to NaN.
- This ensures that:
    - The training set remains unaffected and contains all true values.
    - The validation and test sets can have future values set to NaN without affecting the training data.
- Preserving the true future values in both `validation_true` and `test_true` allows us to evaluate the model properly.

#### **2. Setting Future Values to NaN:**

- We created copies of the validation and test sets (**`validation_true`** and **`test_true`**) before setting NaNs.
- We then set future `soc` and `power` values (from time step 10 onwards) to NaN in the validation and test sets.
- This simulates the scenario where, during validation and testing, we only have the initial 10 time steps and want to predict the next 30.

#### **3. Creating Sequences:**

- In the `create_sequences` function:
    - For **training data**, we use the actual values from the training set for both inputs and outputs.
    - For **validation and test data**, when creating the output sequences (`y`), we retrieve the true future values from `validation_true` and `test_true` (which we preserved earlier).
        - This is important because the `validation` and `test` sets have future values set to NaN.
        - By using the true values, we ensure that `y_validation` and `y_test` contain the correct targets for evaluation.

#### **4. Model Training:**

- The model is trained on `X_train` and `y_train`.
- Validation is performed using `X_validation` and `y_validation`.

#### **5. Model Prediction and Evaluation:**

- Predictions are made on `X_test` using the trained model.
- We compute the Mean Squared Error (MSE) between the predicted values (`y_pred`) and the true values (`y_test_flat`).

---

### **Additional Considerations:**

#### **Normalization of Data:**

- As previously suggested, it's often beneficial to normalize your data, especially when dealing with different scales.
- You can normalize the `soc` and `power` values using `MinMaxScaler` or `StandardScaler`.

**Example:**

```python
from sklearn.preprocessing import MinMaxScaler

# Combine data for fitting scaler
combined_data = pd.concat([train, validation_true, test_true])

# Fit scalers on combined data
soc_scaler = MinMaxScaler()
power_scaler = MinMaxScaler()

# Fit the scalers
soc_columns = [f'soc_{i}' for i in range(40)]
power_columns = [f'power_{i}' for i in range(40)]
soc_scaler.fit(combined_data[soc_columns])
power_scaler.fit(combined_data[power_columns])

# Function to apply scalers
def normalize_data(df):
    df[soc_columns] = soc_scaler.transform(df[soc_columns])
    df[power_columns] = power_scaler.transform(df[power_columns])
    return df

# Normalize datasets (including validation_true and test_true)
train = normalize_data(train)
validation = normalize_data(validation)
test = normalize_data(test)
validation_true = normalize_data(validation_true)
test_true = normalize_data(test_true)
```

- **Remember**:
    - After making predictions, you need to inverse the scaling to get back to the original values before evaluating or plotting.

**Inverse Transform Example:**

```python
# Reshape y_pred_flat to appropriate shape for inverse_transform
y_pred_reshaped = y_pred_flat.reshape(-1, 2)
y_test_reshaped = y_test_flat.reshape(-1, 2)

# Inverse transform soc and power separately
y_pred_soc = soc_scaler.inverse_transform(y_pred_reshaped[:, 0].reshape(-1, 1))
y_pred_power = power_scaler.inverse_transform(y_pred_reshaped[:, 1].reshape(-1, 1))

y_test_soc = soc_scaler.inverse_transform(y_test_reshaped[:, 0].reshape(-1, 1))
y_test_power = power_scaler.inverse_transform(y_test_reshaped[:, 1].reshape(-1, 1))

# Combine back to get final predictions and true values
y_pred_final = np.hstack([y_pred_soc, y_pred_power])
y_test_final = np.hstack([y_test_soc, y_test_power])

# Now compute MSE
mse = mean_squared_error(y_test_final, y_pred_final)
print(f'Test MSE (after inverse transform): {mse}')
```

---

#### **Visualizing Predictions:**

- You can plot the predicted `soc` and `power` against the true values for a sample to see how well your model is performing.

**Example:**

```python
import matplotlib.pyplot as plt

# Choose a sample to visualize
sample_index = 0  # Change as needed

# Get the predicted and true values for the sample
y_pred_sample = y_pred[sample_index]
y_test_sample = y_test[sample_index]

# Inverse transform if data was normalized
y_pred_soc = soc_scaler.inverse_transform(y_pred_sample[:, 0].reshape(-1, 1)).flatten()
y_pred_power = power_scaler.inverse_transform(y_pred_sample[:, 1].reshape(-1, 1)).flatten()

y_test_soc = soc_scaler.inverse_transform(y_test_sample[:, 0].reshape(-1, 1)).flatten()
y_test_power = power_scaler.inverse_transform(y_test_sample[:, 1].reshape(-1, 1)).flatten()

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(30), y_test_soc, label='True SOC')
plt.plot(range(30), y_pred_soc, '--', label='Predicted SOC')
plt.title('SOC Prediction')
plt.xlabel('Time Steps')
plt.ylabel('SOC')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(30), y_test_power, label='True Power')
plt.plot(range(30), y_pred_power, '--', label='Predicted Power')
plt.title('Power Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Power')
plt.legend()

plt.tight_layout()
plt.show()
```

---

### **Summary of Changes:**

- **Order Corrections**:
    - Performed data splitting before setting NaNs.
    - Preserved true future values for validation and test sets for evaluation purposes.
- **Adjustments**:
    - Modified the `create_sequences` function to handle training and non-training data differently.
    - Ensured that `X_validation` and `X_test` have NaNs in the future steps (although in this case, only the first 10 steps are used, so NaNs in future steps don't affect `X`).
- **Evaluation**:
    - Used the preserved true values for evaluation.
    - Ensured proper reshaping and inverse transforming for accurate evaluation.
- **Additional Tips**:
    - Consider setting a random seed (`random_state`) in `train_test_split` for reproducibility.
    - Monitor training and validation loss to prevent overfitting.
    - Experiment with different hyperparameters (e.g., LSTM units, number of layers, etc.) to improve model performance.

---

Let me know if you need further clarification or assistance with any part of the code!