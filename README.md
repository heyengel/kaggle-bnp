## Kaggle BNP Paribas Cardif Claims Management Challenge

Current score: 0.47092 (log loss)

BNP Paribas Cardif provided an anonymized database with two categories of claims. The challenge is to predict the category of a claim based on features available early in the process to help BNP accelerate its claims process and provide better service to its customers.

This dataset contained over 100 features but they were all anonymized into generic column names which made feature creation rather challenging. I used random forest models to get some sense as far as which features were more important and tried creating new features squaring the values or finding interactions between different columns.

Random forest models and parameter grid search was used to predict the target values.

Next steps would be to try boosted trees and SVM.
