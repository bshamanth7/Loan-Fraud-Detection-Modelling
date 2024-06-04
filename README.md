# Loan Fraud Detection Modelling
 Predicting loan defaulters from a list of companies in Small Business Administration (SBA) banks.

![dataset-card](https://github.com/rbhardwaj2186/Loan-Fraud-Detection-Modelling/assets/143745073/f323920a-e89f-4c21-89aa-4de3866686f9)

The Small Business Administration (SBA), established in 1953, has been instrumental in aiding small businesses secure loans. These small businesses are a significant source of employment in the United States, and their growth contributes to economic development. The SBA assists these businesses by guaranteeing bank loans, reducing the risk for banks and encouraging them to lend to small businesses. In case of a loan default, the SBA covers the guaranteed amount, and the bank incurs a loss for the remaining balance.

Despite several success stories like FedEx and Apple, the default rate is quite high. Some economists argue that the banking market functions more efficiently without the SBA’s intervention. However, supporters contend that the social benefits and job creation outweigh the government’s financial costs from defaulted loans.

The data set in question originates from the U.S. SBA loan database. It contains historical data from 1987 through 2014, with 899,164 observations across 27 variables. This data includes information on whether the loan was fully paid off or if the SBA had to charge off any amount, and if so, how much. The utilized data set is a subset of the original one, focusing on loans related to the Real Estate and Rental and Leasing industry in California. This subset contains 2,102 observations and 35 variables. The ‘Default’ column, initially an integer of 1 or zero, was converted to a factor for analysis purposes. If you find this data set useful and choose to download it, an upvote would be appreciated.

# Project 2 Starter

**Here are some tips for submitting your project. You can use the points as partial check list before submission.**

- **Give your notebook a clear and descriptive title.** 
- **Explain your work in Markdown cells.** This will make your notebook easier to read and understand. You can use different colors of font to highlight important points.
- **Remove any unnecessary code or text.** For example, you should not include the template for training and scoring in your final submission.
- **Package your submission in a single file.** I will deduct points for multiple files or incorrect folder structure.
- **Name your notebooks correctly.** Include your name and Net-ID in the file name.
- **Train your TE/WOE encoders on the training set only.** You can train them on the full dataset for your final model.
- **Test your scoring function.** Most students scoring functions in the past din't work, so make sure to test yours before submitting your project.
- **Avoid common mistakes in your scoring function.** For example, your scoring function should not:
  - drop records, expect the target to be passed - check that scoring function returns same number of records as in the validation dataset provided for testing
  - fit TE/WOE/Scalers
  - return anything other than a Pandas DF.
- **Make sure you have the required number of engineered features.** 
- **Don't create features and then not use them in the model**, if there is a reason not to use the feature in the model, explain.
- **Don't include models in your notebook that you didn't train.** This is considered cheating and will result in a grade of zero for the project.
- **Consistently display model performance metrics.** Use AUC for all models and iterations, and don't switch between metrics. For sure don't use accuracy, it is misleading metric for the imbalanced datasets. 
- **Discuss your model results in a Markdown cell.** Don't just print the results; explain what they mean.
- **Include a conclusion section in your notebook.** This is your chance to summarize your findings and discuss the implications of your work.
- **Treat your notebook like a project report that will be read by your manager who can't read Python code.** Make sure your notebook is clear, concise, and easy to understand.
- **Display a preview of your dataset that you used for training.** This will help me understand what features you used in your model.
- **Use the libraries versions specified on eLearning.** For example, you should use H2O 3.42.0.2  
- **Use Python 3.10.11.** If you use another version and your code doesn't work on 3.10.11, it will be considered a bug in your code.
- **When running H2O and want to suppress long prints (for example model summary), include ";" at the end of the command.**
- **Don't include the dataset with your deliverables.**

## Project Requirements Summary

Project 2 is to facilitate students practice of the following Data Science concepts:
- Train and tune classification model
- Perform feature engineering to improve model performance
- Explain/interpret and debug model

- 
## Tasks

The project will include following tasks:
- Load dataset. Don't use "index" column for training.
- Clean up the data:
    - Encode replace missing values
    - Replace features values that appear incorrect
    - Encode numerical variables that come as strings, for example string `$100.01` should be converted to numerical value
- Encode categorical variables
- Split dataset to Train/Test/Validation. If you perform cross-validation while tuning hyper-parameters, you don't need validation dataset.
- Add engineered features. Simple encoding (Target encoding) for individual feature doesn't count to the 10 required engineered features
- Train and tune ML models
- Provide final metrics using Test (hold-out) dataset:
    - Metric to report and optimize for **AUC**
    - Confusion matrix for best F1
- Interpret final trained model using Test dataset:
    - Global feature importance using both Shapley values and permutation feature importance
    - Summary plot with Shapley values 
    - Explain what are the most important features and how they impact model predictions
    - Individual observations analysis using Shapley values. Two records for each of the scenarios with significant probability:
        - Label `0` is correctly identified
        - Label `0` is identified as `1`
        - Label `1` is correctly identified
        - Label `1` is identified as `0`
        - Significant probability means high probability of being correct/in-correct (depending on the scenario)
    - Using residuals analysis identify and report common patterns in the errors made by the model


## Model Training and Tuning

Pick one model to train and tune from the below two options:
- GBM (H2O)
- LightGBM


First, split the dataset to Train/Validation/Test, before applying any encodings clean-up or feature engineering. 
It is important to understand all the steps before model training, so that you can reliably replicate and test them to produce scoring function.

 
