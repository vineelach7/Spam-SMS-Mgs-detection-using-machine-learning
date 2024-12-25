Spam SMS/msgs detection using Machine Learning

## Introduction

This repo shows a machine learning model to classify SMSes **Spam** vs **Ham (non-spam)**. The following code uses various machine learning classification models, including: - Naive Bayes for classification - Logistic Regression- Support Vector Machines.

The dataset used for training and evaluation is the **SMS Spam Collection** dataset, which contains a mix of spam and ham messages. The objective is to train the models and evaluate their performance based on accuracy and other relevant metrics.

## Objective

- **Data Preprocessing**: Clean and preprocess the SMS data, including feature extraction using **TF-IDF**.
- **Training Models**: Train three machine learning models for classification:
  - Naive Bayes (MultinomialNB)
- Logistic Regression
- Support Vector Machines (SVM)
- **Model Evaluation**: Assess each model based on key classification metrics: **Accuracy**, **Precision**, **Recall**, and **F1 Score**.
- **Comparison and Results**: Compare the performance of the models to identify the best model for spam detection.

## Methodology

### Data Cleaning
- The text data is cleaned by removing unwanted characters, symbols, and stopwords.
- **Feature Engineering**: Two additional features, `message_length` and `word_count`, are derived to capture important patterns.
- **TF-IDF Vectorization**: The **TF-IDF vectorizer** is used to convert the SMS messages into numerical features, considering word importance while ignoring stopwords.

### Model Training
1. **Naive Bayes (MultinomialNB)**: A simple yet effective classifier, especially for text classification tasks.
2. **Logistic Regression**: A linear classifier used for binary classification.
3. **Support Vector Machines (SVM)**: A very powerful classifier and works very well in high-dimensional spaces.

### Model Evaluation
All models will be evaluated on the test dataset with the following metrics:
- **Accuracy**: The ratio of correct predictions.
- **Precision**: The ratio of true positives out of all positive predictions.
- **Recall**: The ratio of true positives out of all actual positive samples.
- **F1 Score**: Harmonic mean of precision and recall, balanced measure.

### Results
- After training and testing the models, **Naive Bayes** performs better than **Logistic Regression** and **SVM** on accuracy and overall performance.
- **Naive Bayes**: Best model in this experiment, with the highest **accuracy** at 97%.

## Summary of Results

| Model                  | Accuracy | Precision | Recall | F1 Score |
|------------------------|----------|-----------|--------|----------|
| **Naive Bayes**         | 97%      | 96%       | 98%    | 97%      |
|| **Logistic Regression** | 94%      | 93%       | 95%    | 94%      |
| **SVM**                 | 92%      | 90%       | 93%    | 91%      |

## Challenges Faced
- **Class Imbalance**: The spam and ham messages are distributed unevenly in the dataset, which required careful handling during training.
- **Feature Selection**: Choosing the right features and preprocessing steps was vital to achieve good model performance.
- **Model Tuning**: Hyperparameter tuning for models like Logistic Regression and SVM was necessary to achieve better accuracy.

## Code Structure

```plaintext
/spam-sms-detection
│
├── spam_sms_detection.ipynb    # Jupyter notebook for model training and evaluation
├── spam.csv                    # Dataset (SMS messages)
├── requirements.txt            # List of dependencies
└── README.md                   # Project documentation


## Code Structure

```plaintext
/spam-sms-detection
│
├── spam_sms_detection.ipynb    # Jupyter notebook for model training and evaluation
├── spam.csv                    # Dataset (SMS messages)
├── requirements.txt            # List of dependencies
└── README.md                   # Project documentation

## File Structure

```plaintext
/spam-sms-detection
│
├── spam_sms_detection.ipynb    # Jupyter notebook for model training and evaluation
├── spam.csv                    # Dataset (SMS messages)
├── requirements.txt            # List of dependencies
└── README.md                   # Project documentation
```

- Implemented with an emphasis on precision and accuracy.
- Clean, well-commented code that follows industry standards.
- Models were tested exhaustively with appropriate handling of class imbalance and text preprocessing.

## Installation

### Python Version:
- Python 3.6 or later

### Required Libraries:

To install the requirements, run the following command:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should contain the following:

```text
pandas>=1.1.0
scikit-learn>=0.24.0
matplotlib>=3.2.0
seaborn>=0.11.0
numpy>=1.19.0
```

## Usage

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/spam-sms-detection.git
cd spam-sms-detection
   ```

2. **Download Dataset**:

   - Download the **spam.csv** dataset from [here](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).
   - Place the `spam.csv` file in the repository folder.

3. **Run the Jupyter Notebook**:

   Open and run the `spam_sms_detection.ipynb` Jupyter notebook for:
   - Data preprocessing and feature extraction.
   - Model training and evaluation.
   - Making predictions.

4. **Make Predictions**:

Use the `predict_message()` function to classify new SMS messages:

   ```python
   new_message = "Congratulations! You've won a free ticket to Bahamas. Call now!"
   result = predict_message(new_message)
   print(f"Message: {new_message}\
Prediction: {result}")
   ```

## Documentation

- **Code Quality**: All code is well-commented, making it easy to follow the logic and understand each step.
- **Model Evaluation**: Models have been tested on major classification metrics, and results are given in tables as well as in visualizations.

## Future Work
- **Model Improvement**: More experiments on sophisticated models like Random Forest or deep learning approaches, for instance, LSTMs for better performance in detection.
- **Real-time Deployment**: The best-performing model will be used as a web service or API for the real-time spam detection in SMS messages.

## Contributing

Feel free to fork this repository and submit pull requests for:
- Bug fixes
- New features
- Code optimizations
