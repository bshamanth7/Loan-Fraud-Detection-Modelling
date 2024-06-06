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

### Feature engineering

You should train/fit categorical features encoders on Train only. Use `transform` or equivalent function on Validation/Test datasets.

It is important to understand all the steps before model training, so that you can reliably replicate and test them to produce scoring function.


You should generate various new features. Examples of such features can be seen in the Module-3 lecture on GLMs.  
Your final model should have at least **10** new engineered features.   
On-hot-encoding and target encoding **are not included in the** **10** features to create.    

Ideas for Feature engineering for various types of variables:
1. https://docs.h2o.ai/driverless-ai/1-10-lts/docs/userguide/transformations.html
2. List of important features identified by DriverlessAI AutoML tool

### Model performance in H2O Driverless AI (performance to aim for)

I run experiments in DAI without any data manipulation. It means that potential improvement in performance can be gained with additional feature engineering. 
AUC on hold-out: 0.855

Top features:
- CVTE:Bank:BankState:FranchiseCode:SBA_Appv.0
- CVTE:Bank:BankState:NAICS:UrbanRural.0
- WoE:Bank:BankState:NAICS.0


CVTE: cross-validated target encoding
WoE : weight of evidence 


**Note**: 
- You don't have to perform feature engineering using H2O-3 even if you decided to use H2O-3 GBM for model training.
- It is OK to perform feature engineering using any technique, as long as you can replicate it correctly in the Scoring function.

### Model Tuning

- Hyper-parameter tuning. Your hyper-parameter search space should have at least 50 combinations.
- To avoid overfitting and provide you with reasonable estimate of model performance on hold-out dataset, you will need to split your dataset as following:
    - Train, will be used to train model
    - Validation, will be used to validate model each round of training
    - Testing, will be used to provide final performance metrics, used only once on the final model
- Feature engineering. See project description
- **Use AUC for all models and iterations, and don't switch between metrics.** For sure don't use accuracy, it is misleading metric for the imbalanced datasets.

**Select final model that produces best performance on the Test dataset.**
- For the best model, calculate probability threshold to maximize F1. 
- Report final AUC metric and confusion matrix on the Test dataset using the threshold calculated above.

  ### Threshold calculation

You will need to calculate optimal threshold for class assignment using F1 metric:
- If using sklearn, use F1 `macro`: `f1_score(y_true, y_pred, average='macro')` 
- If using H2O-3, use F1

You will need to find optimal probability threshold for class assignment, the threshold that maximizes above F1.


### Scoring function

The Project will be graded based on the completeness and performance of your final model against the hold-out dataset.
The hold-out dataset provide in the eLearning. As part of your deliverables, you will need to submit a scoring function. 


The scoring function will perform the following:
- Accept dataset in the same format as provided with the project, minus "MIS_Status" column
- Load trained model and any encoders/scalers that are needed to transform data
- Transform dataset into format that can be scored with the trained model
- Score the dataset and return the results, for each record
    - **index** : Record ID
    - **label** : Record label as determined by final model (0 or 1) you need to assign the label based on maximum F1 threshold
    - **probability_0**	: probability of class 0
    - **probability_1** : probability of class 1
    
See full example of scoring function in Project 1 description.
Test your scoring function on the hold-out dataset provided in the eLearning, and validate that it returns the same number of records as in the hold-out dataset.


### Deliverables in a single zip file in the following structure:
- `notebook` (folder)
    - Jupyter notebook with complete code to manipulate data, train and tune final model. `ipynb` format.
    - Jupyter notebook with scoring function. `ipynb` format.
- `artifacts` (folder)
    - Model and any potential encoders in the "pkl" format or native H2O-3 format (for H2O-3 model)
    - Scoring function that will load the final model and encoders. Separate from above notebook or `.py` file



Your notebook should include explanations about your code and be designed to be easily followed and results replicated. Once you are done with the final version, you will need to test it by running all cells from top to bottom after restarting Kernel. It can be done by running `Kernel -> Restart & Run All`


**Important**: To speed up progress, first produce working code using a small subset of the dataset.



