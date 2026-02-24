🚗 Car Price Prediction (Machine Learning)

End-to-end machine learning pipeline for predicting used sports car prices based on a real-world Kaggle dataset.

The project covers data preprocessing, feature engineering, model training, evaluation, and deployment through an interactive Streamlit web application.

⸻

📊 Model Performance (Test Set)
	•	R²: 0.918
	•	MAE: $51,795
	•	RMSE: $222,564
	•	MAPE: 11.16%

The model explains over 91% of the variance in car prices, demonstrating strong predictive performance.

⸻

🔍 Technical Highlights
	•	Cleaned and preprocessed real-world dataset
	•	Feature engineering and transformation
	•	Random Forest regression model
	•	Model evaluation using MAE, RMSE, R², and MAPE
	•	Model artifact persistence using joblib
	•	Interactive dashboard built with Streamlit

⸻

🛠 Tech Stack

Python • Pandas • NumPy • Scikit-learn • Matplotlib • Seaborn • Streamlit

⸻

📂 Project Structure
	•	train_model.py – Model training pipeline
	•	app.py – Streamlit web application
	•	requirements.txt – Project dependencies
	•	model_artifacts/ – Saved model and evaluation outputs

⸻

▶ Run Locally

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

⸻

📈 Deployment

The model is deployed as an interactive Streamlit application that allows users to explore predictions and performance metrics.

