# Sentiment Analysis with Pre-trained Models

## Objective
The objective of this project is to perform sentiment analysis on movie reviews using a pre-trained transformer model and evaluate its performance.

## Data Description
- **Dataset:** IMDB movie reviews dataset
- **Source:** Hugging Face datasets library
- **Sample size:** 500 reviews (subset of the full dataset)
- **Features:**
  - `'text'`: The movie review text
  - `'label'`: Binary sentiment label (0 for negative, 1 for positive)

## Methodology

### Data Preparation
1. Load the IMDB dataset using the Hugging Face datasets library.
2. Clean the text data:
   - Remove HTML tags
   - Remove special characters
   - Remove extra spaces
3. Truncate reviews to fit within the model's maximum token limit (512 tokens).

### Model
- **Pre-trained model:** DistilBERT base model fine-tuned on SST-2 dataset
- **Task:** Sentiment analysis (binary classification)
- **Library:** Hugging Face Transformers

### Analysis Process
1. Apply sentiment analysis to the cleaned and truncated reviews.
2. Process reviews in batches to improve efficiency.
3. Convert model outputs to binary labels (0 for negative, 1 for positive).
4. Split the data into training (80%) and test (20%) sets.

## Model Evaluation
The model's performance is evaluated on the test set using the following metrics:
## Results
- Accuracy: 0.89
- Precision: 0.0
- Recall: 0.0
- F1-score: 0.0


## Discussion
The model achieves a high accuracy of 89%, but the precision, recall, and F1-score are 0. This unusual result suggests that:
1. The model might be predicting all reviews as one class (likely negative, given the dataset's characteristics).
2. There could be an issue with the label encoding or prediction processing.
3. The small sample size (500 reviews) might not be representative of the full dataset.

## Benefits
- Utilizes a pre-trained model, reducing the need for extensive training data and computational resources.
- Provides a quick way to perform sentiment analysis on text data.
- Can be easily integrated into larger natural language processing pipelines.

## Conclusion
While the high accuracy is promising, the other metrics indicate that the model's performance needs improvement. Further investigation into the data distribution, label encoding, and possibly using a larger sample size or the full dataset could lead to more reliable and balanced results.


  
Notebook link[https://colab.research.google.com/drive/1oeT9ssV2fxbdxNuiRzcLvAmq0BAX1F64#scrollTo=aoD4j14qSKer]

# Model 2

# Sentiment Analysis with BERT Neural Network and Python

## Introduction
This project focuses on sentiment analysis using the BERT (Bidirectional Encoder Representations from Transformers) model. BERT is a state-of-the-art transformer-based model developed by Google, designed to understand the context of words in search queries. This project leverages BERT for classifying sentiments in text data, providing a highly accurate and robust solution for natural language processing tasks.

## Data Description
The dataset used in this project consists of text samples labeled with sentiments. The primary columns include:
- **Text**: The input text data.
- **Sentiment**: The target variable indicating the sentiment (e.g., positive, negative, neutral).

## Objectives
The main objectives of this project are:
1. To preprocess the text data for sentiment analysis.
2. To build and fine-tune a BERT model for sentiment classification.
3. To evaluate the performance of the BERT model on the sentiment analysis task.

## Methodology

### 1. Data Preprocessing
The data preprocessing steps involve:
- Tokenizing the text data using BERT's tokenizer.
- Padding and truncating the sequences to ensure uniform input size.
- Encoding the sentiment labels.

### 2. Model Building
The BERT model is fine-tuned for the sentiment classification task. Key steps include:
- Loading the pre-trained BERT model.
- Adding a classification layer on top of the BERT model.
- Training the model on the preprocessed dataset.

### 3. Model Evaluation
The performance of the BERT model is evaluated using metrics such as accuracy, precision, recall, and F1-score. These metrics provide a comprehensive understanding of the model's performance in predicting sentiments.

### Benefits and Applications
The BERT model's ability to understand context and nuances in text makes it highly effective for sentiment analysis. This can be beneficial in various applications, including:
- Social media monitoring: Analyzing public sentiment towards brands, products, or events.
- Customer feedback analysis: Understanding customer opinions and improving services.
- Market research: Gauging public reaction to new products or marketing campaigns.

## Conclusion
This project demonstrates the effectiveness of BERT in performing sentiment analysis. By leveraging BERT's advanced natural language understanding capabilities, we achieve high accuracy in classifying sentiments. The results indicate that BERT is a powerful tool for various NLP tasks, providing valuable insights from text data.

# Model 3

# TV Advertising and Sales Analysis Project

## Project Overview
This project analyzes the relationship between TV advertising spending and sales using a dataset provided in the 'tvmarketing.csv' file. We employ linear regression to predict sales based on TV advertising expenditure and evaluate the model's performance using various metrics.

## Data Description
The dataset contains two main variables:
- TV: TV advertising spending (independent variable)
- Sales: Product sales (dependent variable)

## Methodology

### 1. Data Preparation
- Load the dataset using pandas
- Split the data into features (TV advertising) and target (Sales)
- Divide the data into training and testing sets

### 2. Model Development
- Implement a linear regression model using scikit-learn
- Train the model on the training data

### 3. Prediction
- Use the trained model to make sales predictions on the test data

### 4. Evaluation
We evaluate our model using both regression and classification metrics:

#### Regression Metrics
- Mean Squared Error (MSE)
- R-squared (R²) Score

#### Classification Metrics
To calculate classification metrics, we convert our regression problem into a binary classification:
- Define a threshold (median sales) to classify sales as 'high' or 'low'
- Calculate Accuracy, Precision, Recall, and F1-score

## Results

### Regression Metrics
- Mean Squared Error: 10.20
- R-squared Score: 0.68

### Model Coefficients
- Intercept: 7.12
- Coefficient: 0.05

### Classification Metrics
- Accuracy: 0.85
- Precision: 0.82
- Recall: 0.90
- F1-score: 0.86

## Interpretation of Results

### Regression Analysis
- The R² score of 0.68 indicates that 68% of the variability in sales can be explained by TV advertising spending.
- For every unit increase in TV advertising, sales are predicted to increase by 0.05 units.
- The model predicts a baseline sales of 7.12 units when there is no TV advertising.

### Classification Analysis
- The model correctly classifies 85% of cases as high or low sales.
- When predicting high sales, the model is correct 82% of the time (precision).
- The model identifies 90% of all actual high sales cases (recall).
- The F1-score of 0.86 indicates a good balance between precision and recall.

## Benefits of the Project
1. **Data-Driven Decision Making**: Provides quantitative insights into the relationship between TV advertising and sales.
2. **Performance Prediction**: Enables prediction of sales performance based on advertising spend.
3. **Resource Allocation**: Helps in optimizing advertising budget allocation.
4. **Benchmark Creation**: Establishes a baseline for comparing future marketing strategies.

## Conclusion
The analysis reveals a strong positive relationship between TV advertising spending and sales. The model demonstrates good predictive power, especially in identifying high sales scenarios. However, the R² score suggests that other factors not included in this analysis also influence sales.

## Future Steps
1. Incorporate additional variables that might affect sales.
2. Explore non-linear relationships using more advanced regression techniques.
3. Conduct time series analysis to account for temporal effects on advertising and sales.
4. Implement cross-validation for more robust model evaluation.

## Why These Steps?
1. **Data Preparation**: Ensures clean, properly formatted data for accurate analysis.
2. **Model Development**: Establishes a mathematical relationship between advertising and sales.
3. **Prediction**: Allows testing of the model on unseen data, simulating real-world application.
4. **Evaluation**: Provides multiple perspectives on model performance, ensuring thorough assessment.
5. **Result Interpretation**: Translates statistical findings into actionable business insights.
6. **Future Steps**: Identifies areas for improvement and expansion of the analysis.

By following these steps, we create a comprehensive analysis that not only provides valuable insights into the current relationship between TV advertising and sales but also lays the groundwork for more advanced analyses in the future.
