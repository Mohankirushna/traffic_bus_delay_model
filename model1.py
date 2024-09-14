import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

# Load the data
df = pd.read_csv('/Users/mohankirushna.r/Downloads/traffic_data_two_weeks.csv')

# Convert 'departure_time' to datetime and extract hour and day of the week
df['departure_time'] = pd.to_datetime(df['departure_time'])
df['hour'] = df['departure_time'].dt.hour
df['day_of_week'] = df['departure_time'].dt.dayofweek

# Check for missing values
df.isnull().sum()

# Drop rows with missing values, if any
df.dropna(inplace=True)

# Select the relevant features for training
X = df[['distance_meters', 'duration_in_traffic_seconds', 'hour', 'day_of_week']]
y = df['delay']  # Update the target to predict 'delay'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Calculate Percentage Error and handle division by zero
percentage_errors = np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, np.nan)) * 100

# Exclude NaN values from percentage error calculation
mean_percentage_error = np.nanmean(percentage_errors)
print(f"Mean Percentage Error: {mean_percentage_error}%")

# Calculate R² Score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2}")

# Save the model to a file
joblib.dump(model, 'delay_predictor_model.joblib')

# Create a DataFrame of expected vs predicted results
results_df = pd.DataFrame({'Expected': y_test, 'Predicted': y_pred})

# Tkinter GUI setup
root = tk.Tk()
root.title("Traffic Delay Prediction Visualization")

def plot_graph(graph_type):
    fig, ax = plt.subplots(figsize=(10, 6))

    if graph_type == "Expected vs. Predicted Values":
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Expected Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Expected vs. Predicted Values')
    elif graph_type == "Residuals vs. Predicted Values":
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs. Predicted Values')
    
    ax.grid(True)

    # Clear the previous plot before drawing the new one
    for widget in frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Frame to hold the plot
frame = tk.Frame(root)
frame.pack()

# Dropdown menu to select the graph
options = ["Expected vs. Predicted Values", "Residuals vs. Predicted Values"]
selected_graph = tk.StringVar(value=options[0])

dropdown = ttk.Combobox(root, textvariable=selected_graph, values=options)
dropdown.pack(pady=10)

# Button to plot the selected graph
plot_button = tk.Button(root, text="Plot Graph", command=lambda: plot_graph(selected_graph.get()))
plot_button.pack()

# Start the Tkinter event loop
root.mainloop()
