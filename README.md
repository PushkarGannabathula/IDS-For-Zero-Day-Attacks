🛡️ Intrusion Detection System (IDS) for Zero-Day Attacks 🚀

Welcome to the Intrusion Detection System (IDS) project! This innovative system uses machine learning to detect zero-day attacks by analyzing network traffic flow. Built with four powerful models—CNN, LSTM, XGBoost, and Random Forest—it predicts whether traffic is benign or an attack. The project also features a user-friendly Streamlit frontend with two sections: one for viewing model metrics and another for testing with custom datasets, complete with interactive pie charts! 📊

🌟 Project Overview
Purpose: Detect zero-day attacks by classifying network traffic as benign or malicious.
Models:
Convolutional Neural Network (CNN) 🎨
Long Short-Term Memory (LSTM) ⏳
Extreme Gradient Boosting (XGBoost) ⚡
Random Forest 🌳
Training Data: Leverages traffic flow data to train the models.
Frontend: Streamlit-based interface with two key sections:
Model Metrics: Displays performance stats for all models. 📈
Test Section: Allows users to upload a small dataset and visualize attack/benign traffic percentages using pie charts for each model. 🖼️
🚀 Features
Multi-Model Approach: Combines CNN, LSTM, XGBoost, and Random Forest for robust detection.
Real-Time Testing: Upload your dataset and see instant results with pie charts per model.
Performance Insights: Detailed metrics (accuracy, precision, recall, F1-score) for each model.
User-Friendly: Simple Streamlit interface for easy interaction. 😊
Visualizations: Attractive pie charts showing benign vs. attack traffic distribution. 🎉

🎯 Usage
Model Metrics:

Navigate to the Model Metrics section.
View pre-computed performance metrics (accuracy, precision, etc.) for all four models.
Check the training history and confusion matrices (if available).

Test for Zero-Day Attacks:

Switch to the Test Section.
Upload a small CSV dataset (e.g., 10,000 rows of traffic data with a Label column).
Click "Predict" to see pie charts showing the percentage of benign vs. attack traffic for each model (CNN, LSTM, XGBoost, Random Forest).

📸 Screenshots
Check out the app in action! Below are screenshots of the results:

Model Metrics Section
![IMG-20250630-WA0013](https://github.com/user-attachments/assets/1742f4c9-7971-4849-baa0-07eb009191ec)
![IMG-20250630-WA0014](https://github.com/user-attachments/assets/cf6bc2e8-0940-411a-9937-7c48cbf69287)
![IMG-20250630-WA0015](https://github.com/user-attachments/assets/0af0dddc-ee97-4943-adbf-9f605981478c)

View the performance metrics for all four models in one place!

Test Section with Pie Charts

For Benign Dataset
![IMG-20250630-WA0011](https://github.com/user-attachments/assets/97965fe9-0b75-437f-8025-6657a9365129)
![IMG-20250630-WA0012](https://github.com/user-attachments/assets/9efcfa48-96b0-4035-b750-2d57e5ce247e)

For Standard Dataset
![IMG-20250630-WA0016](https://github.com/user-attachments/assets/56fce7b5-ce8c-42ed-9ae6-f4bc830a2595)
![IMG-20250630-WA0017](https://github.com/user-attachments/assets/5defebe4-a7d9-46cc-9fdf-160a6814637f)

For High Attack Dataset
![IMG-20250630-WA0018](https://github.com/user-attachments/assets/dcd40d61-0d1f-4dd9-91f7-910ca13380af)
![IMG-20250630-WA0019](https://github.com/user-attachments/assets/531f293c-6520-4e57-8816-7ae75f6e9f98)

Upload a dataset and see pie charts for CNN, LSTM, XGBoost, and Random Forest predictions!.

🛠️ How to Set Up
Prerequisites
Python 3.8+
Required libraries: streamlit, numpy, pandas, scikit-learn, tensorflow, xgboost, matplotlib, seaborn, streamlit-option-menu

🙌 Acknowledgments
Thanks to the CICIDS2017 dataset for providing the traffic flow data! 🌐
Inspired by the Streamlit community for the amazing frontend framework. 💻
