# Student Score Predictor

A simple machine learning project that predicts an exam score based on
the number of hours studied. It includes a Streamlit web app for entering
study hours and viewing a prediction.

## How it works

1. Load a sample dataset of study hours and exam scores.
2. Split the data into training and test sets.
3. Train a linear regression model using scikit-learn.
4. Evaluate the model using mean absolute error.
5. Save the trained model and use it in a Streamlit app.

## Technologies

- Python
- pandas
- scikit-learn
- Streamlit
- joblib

## Run the project

Create and activate a Python virtual environment, then install the packages:

```bash
pip install pandas scikit-learn streamlit joblib
```

Train the model:

```bash
python train_model.py
```

Start the app:

```bash
python -m streamlit run app.py
```

## Dataset and limitations

The dataset is small and made up for demonstration purposes. Predictions
are not based on real student data and should not be used to assess actual
students. The reported test error reflects only this sample dataset.
