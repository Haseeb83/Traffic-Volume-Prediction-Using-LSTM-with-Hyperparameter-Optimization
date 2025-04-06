# Traffic-Volume-Prediction-Using-LSTM-with-Hyperparameter-Optimization

Project Description: Traffic Volume Prediction Using LSTM with Hyperparameter Optimization
Overview
The Traffic Volume Prediction project aims to develop a predictive model that forecasts daily traffic counts for various locations (referred to as "aliases") in a city, such as Turingekorset, Birkakorset, Nedre Torekällsgatan, and Nygatan. The model leverages historical and real-time traffic data to predict future traffic volumes, which can assist in urban planning, traffic management, and infrastructure optimization. The project uses a deep learning approach with Long Short-Term Memory (LSTM) networks, optimized through hyperparameter tuning, to capture temporal patterns in traffic data. The results are visualized and evaluated to assess the model’s performance across all locations.

Objectives
Predict Traffic Volumes: Build a model to accurately forecast daily traffic counts for multiple locations using time series data.
Optimize Model Performance: Use hyperparameter optimization to fine-tune the LSTM model for better prediction accuracy.
Evaluate Model Performance: Compute combined Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) across all locations to assess the model’s overall performance.
Visualize Results: Provide intuitive visualizations to compare actual vs. predicted traffic counts and display model performance metrics.
Dataset
Source: The dataset consists of traffic data stored in JSON files, collected from various locations (aliases) in a city.
Historical data (e.g., Nygatan from October 2023) and recent data (e.g., Turingekorset, Birkakorset, and Nedre Torekällsgatan from November to December 2024) are included.
Structure: Each JSON file contains traffic counts with timestamps and vehicle type breakdowns (e.g., Today_Car_A, Today_Bus_B, Today_Truck_C, Today_MBike_D).
If vehicle type columns are missing, alternative numeric columns (e.g., volume) are used to compute the total traffic count.
Preprocessing:
Data is aggregated at a daily granularity.
Total traffic counts (total_count) are calculated by summing vehicle type counts (or using alternative columns if needed).
Additional features like day_of_week and hour are extracted to capture temporal patterns.
Data is scaled using MinMaxScaler to a range of [0, 1] for model training.
Methodology
The project follows a structured pipeline for data processing, model development, evaluation, and visualization:

Data Loading and Preprocessing:
JSON files are loaded and processed into Pandas DataFrames.
Timestamps are converted to datetime format, and features like day_of_week and hour are extracted.
Traffic counts are aggregated into a total_count column, with outlier capping at the 99th percentile.
Data is scaled using MinMaxScaler to prepare it for the LSTM model.
Time Series Preparation:
The data is transformed into sequences of length time_steps (set to 7 days) to capture temporal dependencies.
Each sequence includes features (total_count, day_of_week, hour) and predicts the next day’s traffic count.
Model Development:
An LSTM model is built using TensorFlow/Keras, consisting of multiple LSTM layers, dropout layers for regularization, and dense layers for output.
Hyperparameters (e.g., number of LSTM units, learning rate, batch size) are optimized using Optuna to minimize validation loss.
Training and Prediction:
The dataset is split into training and validation sets (80-20 split).
The LSTM model is trained on the training set for each alias, with early stopping to prevent overfitting.
Predictions are made on the entire dataset (no date range restriction) for evaluation.
Evaluation:
Actual and predicted traffic counts are collected across all aliases into a single dataset.
Combined MSE and RMSE are computed to evaluate the model’s overall performance:
MSE: Measures the average squared error, emphasizing larger errors.
RMSE: The square root of MSE, providing an interpretable error in the same units as the traffic counts (vehicles).
Visualization:
A line graph compares actual vs. predicted traffic counts for all aliases over a specific period (December 1–15, 2024), with MAE and RMSE displayed in the legend for each alias.
A bar chart visualizes the combined MSE and RMSE for the overall dataset, with numerical values included in the title.
Tools and Frameworks
Python: Core programming language.
Pandas/NumPy: Data manipulation and numerical computations.
TensorFlow/Keras: Building and training the LSTM model.
Optuna: Hyperparameter optimization.
Scikit-learn: Data scaling (MinMaxScaler) and evaluation metrics (mean_squared_error).
Plotly: Interactive visualizations (line graphs and bar charts).
Datetime, os, glob, json: Handling timestamps, file system operations, and data loading.
Implementation Steps
Data Loading:
Read JSON files from a specified directory using glob and json.
Categorize data into traffic_data (recent) and historical_data based on file names and data volume.
Data Preprocessing:
Convert raw data into a structured DataFrame using prepare_traffic_df.
Compute total_count by summing vehicle type counts or using alternative columns.
Extract features (day_of_week, hour) and scale the data.
Model Training:
For each alias, prepare time series sequences using prepare_time_series_data.
Use Optuna to optimize LSTM hyperparameters (objective function minimizes validation loss).
Train the LSTM model with early stopping and save the best model for each alias in models and scalers.
Prediction and Evaluation:
Generate predictions for the entire dataset using evaluate_predictions_entire_dataset.
Concatenate actual (y_true_inv) and predicted (y_pred_inv) values across all aliases.
Compute combined MSE and RMSE for the overall dataset.
Visualization:
Section #2: A single line graph shows actual vs. predicted traffic for all aliases (Dec 1–15, 2024), with per-alias MAE/RMSE in the legend.
Section #3: A bar chart displays the combined MSE and RMSE for the overall dataset, with values in the title.
Evaluation Metrics
Mean Squared Error (MSE): Measures the average squared difference between actual and predicted traffic counts, emphasizing larger errors.
Root Mean Squared Error (RMSE): The square root of MSE, providing the average error in the same units as the traffic counts (vehicles).
These metrics are computed over the entire dataset across all aliases, giving a holistic view of the model’s performance.
Outcomes
Accurate Predictions: The LSTM model, optimized with Optuna, successfully captures temporal patterns in traffic data, providing reasonable predictions for daily traffic counts.
Combined Performance Metrics: The overall MSE and RMSE provide a single measure of the model’s accuracy across all locations, making it easy to assess performance.
Insightful Visualizations:
The line graph allows for a direct comparison of actual vs. predicted traffic counts, with per-alias MAE/RMSE for detailed analysis.
The bar chart succinctly visualizes the combined MSE and RMSE, highlighting the model’s overall error.
Scalability: The model can handle multiple aliases and is adaptable to new locations by retraining on additional data.
Challenges and Solutions
Missing Traffic Counts: Some aliases lacked Today_* columns for vehicle types. This was resolved by using alternative numeric columns (e.g., volume) to compute total_count.
Hyperparameter Tuning: Initial models had suboptimal performance. Optuna was used to systematically tune LSTM hyperparameters, improving prediction accuracy.
Visualization Clutter: Early versions generated separate graphs for each alias. This was addressed by combining all aliases into a single graph with clear legend annotations.
Future Improvements
Additional Features: Incorporate external factors like weather conditions, holidays, or events to improve prediction accuracy.
Model Alternatives: Experiment with other time series models, such as Transformer-based models (e.g., Temporal Fusion Transformer) or Prophet, for comparison.
Real-Time Prediction: Deploy the model in a real-time setting to provide live traffic forecasts for city planners.
Error Analysis: Perform a deeper analysis of prediction errors (e.g., by time of day or day of week) to identify areas for improvement.
Conclusion
The Traffic Volume Prediction project successfully demonstrates the application of LSTM networks for time series forecasting in an urban traffic context. By leveraging historical and recent traffic data, the model provides accurate predictions for multiple locations, with performance optimized through hyperparameter tuning. The combined MSE and RMSE metrics offer a clear evaluation of the model’s overall accuracy, while interactive visualizations provide insights into the predictions. This project lays a strong foundation for traffic management applications and can be extended with additional features and models for even better performance.
