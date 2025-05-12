# Comparative Analysis of Machine Learning Models for Airline Reviews Classification

## Project Overview

This project implements and evaluates multiple machine learning algorithms to perform sentiment analysis on airline reviews. The system classifies customer feedback to help airlines understand public perception and identify areas for service improvement.

## Objective

The primary goal is to develop robust models capable of accurately classifying airline reviews based on textual content and associated features, providing actionable insights to enhance customer service and satisfaction in the aviation industry.

## Dataset

The dataset comprises airline reviews with the following features:
- Title
- Name
- Review Date
- Airline
- Verified status
- Reviews (text content)
- Type of Traveller
- Month Flown
- Route
- Class
- Numerical ratings (Seat Comfort, Staff Service, Food & Beverages, Inflight Entertainment, Value For Money, Overall Rating)
- Recommendation status (target variable)

## Methodology

### Data Preprocessing
- Text cleaning and normalization
- Feature extraction using TF-IDF vectorization
- One-hot encoding of categorical variables
- Handling missing values
- Feature combination (numerical, categorical, and text features)

### Models Implemented
1. **Support Vector Machine (SVM)** - Optimizes the margin between classes while determining the best hyperplane to divide various classes
2. **K-Nearest Neighbors (KNN)** - Classifies based on majority class among K closest neighbors
3. **Naïve Bayes** - Probabilistic classifier based on Bayes' theorem with feature independence assumption
4. **Artificial Neural Network (ANN)** - Multi-layer network with dense layers for complex pattern recognition
5. **Random Forest** - Ensemble method combining multiple decision trees to improve generalization

## Results

Performance metrics for each model:

| Model | Accuracy | Precision | Recall | F1 Score | Mean Absolute Error |
|-------|----------|-----------|--------|----------|---------------------|
| SVM | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| KNN | 0.984 | 0.985 | 0.984 | 0.985 | 0.016 |
| Naive Bayes | 0.996 | 0.993 | 1.0 | 0.997 | 0.004 |
| Artificial NN | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| Random Forest | 0.995 | 0.993 | 0.998 | 0.995 | 0.005 |

## Key Findings

- SVM and ANN achieved perfect classification with 100% accuracy
- All models performed exceptionally well, with accuracy rates above 98%
- The models successfully captured the sentiment patterns in airline reviews
- Feature combination (text, categorical, and numerical) proved effective for classification

## Technologies Used

- Python
- Pandas for data manipulation
- Scikit-learn for machine learning models
- TensorFlow/Keras for neural network implementation
- NLTK for text processing
- Matplotlib for visualization

## Installation and Usage

