## Diabetes-Prediction
This repo includes the diabetes solution of diabetes problem and deploying model using FastAPI (basic usage)


------------


## 1. General Overview of Dataset
- At this problem the **target variable** (which is going to be predicted) is**'Outcome'**column and our features are rest of the dataset. Outcome has classified as *binary*.

- When we analyzing the descriptive statistics of this dataset, we can clearly observe that there are many outliers and illogical entries like **Blood Pressure = 0.**

- Numbers of outcomes seems **Imbalanced**
		"number_of_ones":268
		"number_of_zeros":500
- There is no **NaN (Not a Number)**
- Dtypes are fine.
----
## 2.Approaches and Preprocessing
- Most of tutorials which are on the kaggle and the internet are generally handling the outliers and filling with medians of the feature which includes outlier. Only appyling this technique is increases accuracy scores about **3-4%**.

- In my opinion, to get high accuracies, In addition to above, this dataset must be manipulated with some rules and we must handle imbalance situation that  emphasized above.

- I did handling outlier operations little bit different from tutorials. Difference is I have filled outliers with medians which are according to outcomes. You can find the functions in my jupyter notebook.

- To handle imbalanced dataset, I did oversampling with SMOTE technique from imblearn library. You can find what smote does [here](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/ "here").
- Manipulacting process can be found in **diabetes_project.ipynb** notebook, process has done by results from scatterplots.
---
## 3.Model Selection and Training
- I have trained and tested 4 different classification algorithms before and after oversampling. I can clearly say that oversampling with SMOTE sufficiently increases accuracies both test and train.

- Test size of train/test split  :** 0.2**

> **SVC, BaggingClassifier, RandomForestClassifier, LGBM** has trained and tested.

- ### Results before OverSampling
 - SVC and RandomForest Classifiers are facing with **overfitting problem** *(gap occured between test and train accuracies.).*
 - LGBM ,BaggingClassifier and RandomForestClassifier test accuracies is around **88%**
- ### Results after OverSampling
 - Overfitting problem still occurs.
 - Test accuracies of trees increased to %91-93
 - Accuracy of SVC has increased **78%** to **85%**
 
- From all the reasons above, we can use **LGBM** and **BaggingClassifier**

**P.S. : ** Hyperparameter Optimization process can be weak because of my local computer system. You can try in Cloud.



