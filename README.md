# Diabetes Prediction System

This program is designed to predict whether a patient has diabetes based on various medical attributes. The prediction is made using a Support Vector Machine (SVM) classifier.

## Overview

The dataset used for training the model is in CSV format and contains the following features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (0 for non-diabetic, 1 for diabetic)

## Steps in the Program

1. **Importing Libraries**: The program imports necessary libraries including NumPy, Pandas, and Scikit-learn.

2. **Data Collection and Analysis**:
   - Load the dataset into a Pandas DataFrame.
   - Display the first 5 rows of the dataset.
   - Print the shape of the dataset.
   - Display statistical measures of the dataset.
   - Show the distribution of diabetic and non-diabetic patients.
   - Group the data by the 'Outcome' column and display the mean values for each group.

3. **Data Preparation**:
   - Separate features (X) and labels (Y).
   - Standardize the feature data using `StandardScaler`.

4. **Data Splitting**:
   - Split the data into training and testing sets using an 80-20 split.

5. **Model Training**:
   - Train an SVM classifier with a linear kernel on the training data.

6. **Model Evaluation**:
   - Calculate and print the accuracy score for both training and test data.

7. **Predictive System**:
   - Demonstrate a predictive system where input data for a specific patient is transformed and used to predict whether they have diabetes.

## Usage

To run the program, execute the following steps:

1. Ensure you have all the required libraries installed:
   ```bash
   pip install numpy pandas scikit-learn


diabetes_dataset = pd.read_csv("/path/to/your/diabetes.csv")

*Heres how to make a prediction
```
input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)
input_data_as_np_array = np.asarray(input_data)
input_data_reshaped = input_data_as_np_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)
```
-And here is the prediction results
```
if (prediction[0] == 0):
    print("This person does not have diabetes")
else:
    print("This person has diabetes")
```

## Conclusion

This program successfully trains an SVM classifier to predict diabetes with a reasonable accuracy. It demonstrates the entire process from data loading, preprocessing, training, evaluation, and making predictions.

For more of my projects and achievements, please visit my GitHub and LinkedIn.



