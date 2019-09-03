# hands-on-machine-learning
# FACTORS INFLUENCING STROKE

This notebook combines python and R codes. I prefer doing things like data wrangling in R than python. 


```python
import os          
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
#import sklearn.linear_model as lm
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, precision_recall_curve
from sklearn.metrics import recall_score, precision_recall_curve, roc_auc_score, f1_score, roc_curve, auc
from xgboost import XGBClassifier, plot_importance
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import category_encoders as ce



%matplotlib inline 
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


```


```python
%load_ext rpy2.ipython
```


```r
%%R
library(tidyverse)
library(cowplot)
```

    R[write to console]: Registered S3 methods overwritten by 'ggplot2':
      method         from 
      [.quosures     rlang
      c.quosures     rlang
      print.quosures rlang
    
    R[write to console]: Registered S3 method overwritten by 'rvest':
      method            from
      read_xml.response xml2
    
    R[write to console]: ââ [1mAttaching packages[22m âââââââââââââââââââââââââââââââââââââââ tidyverse 1.2.1 ââ
    
    R[write to console]: [32mâ[39m [34mggplot2[39m 3.1.1       [32mâ[39m [34mpurrr  [39m 0.3.2  
    [32mâ[39m [34mtibble [39m 2.1.1       [32mâ[39m [34mdplyr  [39m 0.8.0.[31m1[39m
    [32mâ[39m [34mtidyr  [39m 0.8.3       [32mâ[39m [34mstringr[39m 1.4.0  
    [32mâ[39m [34mreadr  [39m 1.3.1       [32mâ[39m [34mforcats[39m 0.4.0  
    
    R[write to console]: ââ [1mConflicts[22m ââââââââââââââââââââââââââââââââââââââââââ tidyverse_conflicts() ââ
    [31mâ[39m [34mdplyr[39m::[32mfilter()[39m masks [34mstats[39m::filter()
    [31mâ[39m [34mdplyr[39m::[32mlag()[39m    masks [34mstats[39m::lag()
    
    R[write to console]: 
    ********************************************************
    
    R[write to console]: Note: As of version 1.0.0, cowplot does not change the
    
    R[write to console]:   default ggplot2 theme anymore. To recover the previous
    
    R[write to console]:   behavior, execute:
      theme_set(theme_cowplot())
    
    R[write to console]: ********************************************************
    
    


## Import and investigating the training data


```r
%%R 
train = read.csv("./healthcare-dataset-stroke-data/train_2v.csv", stringsAsFactors=FALSE)
summary(train)
```

           id           gender               age         hypertension    
     Min.   :    1   Length:43400       Min.   : 0.08   Min.   :0.00000  
     1st Qu.:18038   Class :character   1st Qu.:24.00   1st Qu.:0.00000  
     Median :36352   Mode  :character   Median :44.00   Median :0.00000  
     Mean   :36326                      Mean   :42.22   Mean   :0.09357  
     3rd Qu.:54514                      3rd Qu.:60.00   3rd Qu.:0.00000  
     Max.   :72943                      Max.   :82.00   Max.   :1.00000  
                                                                         
     heart_disease     ever_married        work_type         Residence_type    
     Min.   :0.00000   Length:43400       Length:43400       Length:43400      
     1st Qu.:0.00000   Class :character   Class :character   Class :character  
     Median :0.00000   Mode  :character   Mode  :character   Mode  :character  
     Mean   :0.04751                                                           
     3rd Qu.:0.00000                                                           
     Max.   :1.00000                                                           
                                                                               
     avg_glucose_level      bmi       smoking_status         stroke       
     Min.   : 55.00    Min.   :10.1   Length:43400       Min.   :0.00000  
     1st Qu.: 77.54    1st Qu.:23.2   Class :character   1st Qu.:0.00000  
     Median : 91.58    Median :27.7   Mode  :character   Median :0.00000  
     Mean   :104.48    Mean   :28.6                      Mean   :0.01804  
     3rd Qu.:112.07    3rd Qu.:32.9                      3rd Qu.:0.00000  
     Max.   :291.05    Max.   :97.6                      Max.   :1.00000  
                       NA's   :1462                                       



```python
train = %R train
print("The dimenssion of the training set is ", train.shape)
train.head()
```

    The dimenssion of the training set is  (43400, 12)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>30669</td>
      <td>Male</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>children</td>
      <td>Rural</td>
      <td>95.12</td>
      <td>18.0</td>
      <td></td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30468</td>
      <td>Male</td>
      <td>58.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>87.96</td>
      <td>39.2</td>
      <td>never smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16523</td>
      <td>Female</td>
      <td>8.0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>Private</td>
      <td>Urban</td>
      <td>110.89</td>
      <td>17.6</td>
      <td></td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>56543</td>
      <td>Female</td>
      <td>70.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>69.04</td>
      <td>35.9</td>
      <td>formerly smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>46136</td>
      <td>Male</td>
      <td>14.0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>Never_worked</td>
      <td>Rural</td>
      <td>161.28</td>
      <td>19.1</td>
      <td></td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 43400 entries, 1 to 43400
    Data columns (total 12 columns):
    id                   43400 non-null int32
    gender               43400 non-null object
    age                  43400 non-null float64
    hypertension         43400 non-null int32
    heart_disease        43400 non-null int32
    ever_married         43400 non-null object
    work_type            43400 non-null object
    Residence_type       43400 non-null object
    avg_glucose_level    43400 non-null float64
    bmi                  41938 non-null float64
    smoking_status       43400 non-null object
    stroke               43400 non-null int32
    dtypes: float64(3), int32(4), object(5)
    memory usage: 3.6+ MB


## Data Visualisation


```python
scatter_matrix(train.drop("id",axis=1), figsize=(12, 8))
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa2674208>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa38656d8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa381b5c0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa37cc710>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa3780b70>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa37bb160>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa376a710>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa371dc18>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa371dc50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa3689780>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa36bcd30>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa3679320>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa36268d0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa35d8e80>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa3597470>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa3544a20>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa3577fd0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa35335c0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa34e4b70>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa34a2160>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa3451710>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa2613cc0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa25d02b0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa25ff860>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa25b2e10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa256e400>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa251d9b0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa24d1f60>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa250d550>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa24bcb00>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa24790f0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa242b6a0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa23dbc50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa2399240>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa23ca7f0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2fa237ada0>]],
          dtype=object)




![png](output_10_1.png)


Notice a few things in the histograms (on the diagonal):
1. First, the stroke attribute shows that a large number of indivuduals sampled were not affected by stroke. Only aproximatively 2% were affected by stroke.
2. Attributes have different scales. 
3. Historgrams such as that of average glucose level's attribute is tail heavy: it extend much farther to the right of the median thant to the left. This may make it a vit harder for some ML algotithms to detect patterns.


```r
%%R

age_plot <- train%>%ggplot(aes(as.factor(stroke),age)) + geom_boxplot() + theme_bw() + scale_x_discrete(breaks=c(0, 1), labels=c('No', "Yes")) + xlab("Stroke")
bmi_plot <- train%>%ggplot(aes(as.factor(stroke),bmi)) + geom_boxplot() + theme_bw() + scale_x_discrete(breaks=c(0, 1), labels=c('No', "Yes")) + xlab("Stroke")
avg_glu_plot <- train%>%ggplot(aes(as.factor(stroke),avg_glucose_level)) + geom_boxplot() + theme_bw() + scale_x_discrete(breaks=c(0, 1), labels=c('No', "Yes")) + xlab("Stroke")
plot_grid(age_plot, bmi_plot, avg_glu_plot, labels = "AUTO")
```


![png](output_12_0.png)


A quick look at these boxplots reveals that stroke is likekely to affect older people i.e. age seems to influence stroke. Both bmi and avg_glucose_level attribute however do not seem to have a significant influence on stroke occurence as both median (No and Yes) do not appear to be statistically different. 


```r
%%R

age_smo <- ggplot(train, aes(x=as.factor(stroke), y=age, color=as.factor(smoking_status))) + geom_boxplot()+ 
    facet_grid(hypertension ~ heart_disease,labeller = labeller(.rows = label_both, .cols = label_both))+
    theme_bw() + scale_x_discrete(breaks=c(0, 1), labels=c('No', "Yes")) + xlab("Stroke")
age_smo
```


![png](output_14_0.png)


Upper left panel shows that smoking, heart disease and hypertension do not significantly influence the occurance of stroke as among those affected by stroke. Among the subset affected by stroke, the median of those never smoked is comparable with those formerly smoked but suprisely, exceeds that of those who smoke. This is probably due to the missing information on the smoking_status attribute.


```r
%%R

age_maried <- ggplot(train, aes(x=as.factor(stroke), y=age, color=as.factor(ever_married))) + geom_boxplot()+ 
    facet_grid(hypertension ~ heart_disease,labeller = labeller(.rows = label_both, .cols = label_both))+
    theme_bw() + scale_x_discrete(breaks=c(0, 1), labels=c('No', "Yes")) + xlab("Stroke")
    
age_maried
```


![png](output_16_0.png)



```r
%%R

age_wor_typ <- ggplot(train, aes(x=as.factor(stroke), y=age, color=as.factor(work_type))) + geom_boxplot()+ 
    facet_grid(hypertension ~ heart_disease,labeller = labeller(.rows = label_both, .cols = label_both))+
    theme_bw() + scale_x_discrete(breaks=c(0, 1), labels=c('No', "Yes")) + xlab("Stroke")
age_wor_typ
```


![png](output_17_0.png)



```r
%%R

age_resi_typ <- ggplot(train, aes(x=as.factor(stroke), y=age, color=as.factor(Residence_type))) + geom_boxplot()+ 
    facet_grid(hypertension ~ heart_disease,labeller = labeller(.rows = label_both, .cols = label_both))+
    theme_bw() + scale_x_discrete(breaks=c(0, 1), labels=c('No', "Yes")) + xlab("Stroke")
age_resi_typ
```


![png](output_18_0.png)



```r
%%R

age_gender <- ggplot(train, aes(x=as.factor(stroke), y=age, color=as.factor(gender))) + geom_boxplot()+ 
    facet_grid(hypertension ~ heart_disease,labeller = labeller(.rows = label_both, .cols = label_both))+
    theme_bw() + scale_x_discrete(breaks=c(0, 1), labels=c('No', "Yes")) + xlab("Stroke")
age_gender
```


![png](output_19_0.png)


## Preparing the data for ML algorithm
- Checking for missing values


```python
missing_val_prop_by_column = (train.isnull().sum()/train.shape[0])
print(missing_val_prop_by_column[missing_val_prop_by_column > 0])
missing_val_prop_by_column.index
```

    bmi    0.033687
    dtype: float64





    Index(['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
           'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
           'smoking_status', 'stroke'],
          dtype='object')



Athough smoking is a major risk factors for stroke, more than 20% of the corresponding values in the data here are missing. In the absence of any information, I assume that under 18 years old are classified as *never somoke*, 18-45 smokes and >45 formerly smoked. A thorough analysis could be considered using information from different sources.


```r
%%R
fill_missing_smoking_status <- function(age,smoking_status){
    
     smoking_status<- replace(smoking_status, age<18 & smoking_status=="", "never smoke")
     smoking_status<- replace(smoking_status, age>=18 & age <45 & smoking_status=="", "smokes")
     smoking_status<- replace(smoking_status, age>= 45 & smoking_status=="", "formerly smoked")
     return(smoking_status)
}
train_update <- train%>%dplyr::mutate(smoking_status=fill_missing_smoking_status(age,smoking_status) )%>%
        as.data.frame()
head(train_update)
```

         id gender age hypertension heart_disease ever_married    work_type
    1 30669   Male   3            0             0           No     children
    2 30468   Male  58            1             0          Yes      Private
    3 16523 Female   8            0             0           No      Private
    4 56543 Female  70            0             0          Yes      Private
    5 46136   Male  14            0             0           No Never_worked
    6 32257 Female  47            0             0          Yes      Private
      Residence_type avg_glucose_level  bmi  smoking_status stroke
    1          Rural             95.12 18.0     never smoke      0
    2          Urban             87.96 39.2    never smoked      0
    3          Urban            110.89 17.6     never smoke      0
    4          Rural             69.04 35.9 formerly smoked      0
    5          Rural            161.28 19.1     never smoke      0
    6          Urban            210.95 50.1 formerly smoked      0



```python
train_update = %R train_update
stroke_labels = train_update["stroke"]
stroke_predictors = train_update.drop(["stroke","id"], axis=1)
```

Remove columns with more than 10 factors (for convenience, if they exist, to avoid encreasing too much the size of variables after OneHotEncoding) and selecting numeric variables


```python
low_cardinality_cols = [cname for cname in stroke_predictors.columns if 
                                stroke_predictors[cname].nunique() < 10 and
                                stroke_predictors[cname].dtype == "object"]
numeric_cols = [cname for cname in stroke_predictors.columns if 
                                stroke_predictors[cname].dtype in ['int64', 'float64']]
```

## Transformation Pipelines
Recall that the body mass index (bmi) is the only numerical attribute with missing values. I use median to fill in these gaps. The 


```python
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scaler", StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, numeric_cols),
    ("cat", OneHotEncoder(), low_cardinality_cols),
])

stroke_prepared = full_pipeline.fit_transform(train_update[low_cardinality_cols + numeric_cols])
stroke_prepared.shape
```




    (43400, 19)



## Generate samples to account for the issue of imbalanced class


```python
class OverUnderSampling(BaseEstimator, TransformerMixin):
    """
    A short transformer to generate new samples 
    in the classes which are under-represented/over-represented.
    We use randomly sampling with replacement and 
    Synthetic Minority Oversampling Technique (SMOTE).
    """
    def __init__(self, smote = True, over = True):
        self.smote = smote
        self.over = over
    def fit(sel, X, y=None):
        return self
    def transform(self, X, y=None):
        if self.over:
            if self.smote:
                X_resampled, y_resampled = SMOTE().fit_resample(X, y)
                return X_resampled, y_resampled
            else:
                ros = RandomOverSampler(random_state=0)
                X_resampled, y_resampled = ros.fit_resample(X,y)
                return X_resampled, y_resampled
        else:
            rus = RandomUnderSampler(random_state=0, replacement=True)
            X_resampled, y_resampled = rus.fit_resample(X, y)
            return X_resampled, y_resampled
            
```


```python
 over_samp = OverUnderSampling()
 X, y = over_samp.transform(stroke_prepared, stroke_labels)
```

## Splitting the data into training and validation sets


```python
X, X_val, y, y_val = train_test_split(stroke_prepared, stroke_labels, test_size=0.33, random_state=42)
type(X_val)
```




    numpy.ndarray




```r
%%R -i X
xx <- X
X1 <- as.data.frame(xx)
```


```r
%%R -i X_val
yy <- X_val
X2 <- as.data.frame(yy)
```


```python
X = %R X1
X_val = %R X2
X_train = X
y_train = y
```

**Using techniques to deal with imbalanced data developped [here](https://www.kaggle.com/npramod/techniques-to-deal-with-imbalanced-data):**


```python
def benchmark(sampling_type,X,y):
    lr = LogisticRegression(penalty = 'l1')
    param_grid = {'C':[0.01,0.1,1,10]}
    gs = GridSearchCV(estimator=lr, param_grid=param_grid, scoring='accuracy', cv=5, verbose=2)
    gs = gs.fit(X.values,y.values.ravel())
    return sampling_type,gs.best_score_,gs.best_params_['C']

def transform(transformer,X,y):
    print("Transforming {}".format(transformer.__class__.__name__))
    X_resampled,y_resampled = transformer.fit_sample(X.values,y.values.ravel())
    return transformer.__class__.__name__,pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)
```


```python
datasets = []
datasets.append(("base",X_train,y_train))
datasets.append(transform(SMOTE(n_jobs=-1),X_train,y_train))
datasets.append(transform(RandomOverSampler(),X,y))
datasets.append(transform(RandomUnderSampler(),X_train,y_train))
```

    Transforming SMOTE
    Transforming RandomOverSampler
    Transforming RandomUnderSampler



```python
# benchmark_scores = []
# for sample_type,X,y in datasets:
#     print('______________________________________________________________')
#     print('{}'.format(sample_type))
#     benchmark_scores.append(benchmark(sample_type,X,y))
#     print('______________________________________________________________')
```


```python
# scores = []
# train models based on benchmark params
# for sampling_type,score,param in benchmark_scores:
#     print("Training on {}".format(sampling_type))
#     lr = LogisticRegression(penalty = 'l1',C=param)
#     for s_type,X,y in datasets:
#         if s_type == sampling_type:
#             lr.fit(X.values,y.values.ravel())
#             pred_test = lr.predict(X_val.values)
#             pred_test_probs = lr.predict_proba(X_val.values)
#             probs = lr.decision_function(X_val.values)
#             fpr, tpr, thresholds = roc_curve(y_val.values.ravel(),pred_test)
#             p,r,t = precision_recall_curve(y_val.values.ravel(),probs)
#             scores.append((sampling_type,
#                            f1_score(y_val.values.ravel(),pred_test),
#                            precision_score(y_val.values.ravel(),pred_test),
#                            recall_score(y_val.values.ravel(),pred_test),
#                            accuracy_score(y_val.values.ravel(),pred_test),
#                            auc(fpr, tpr),
#                            auc(p,r,reorder=True),
#                            confusion_matrix(y_val.values.ravel(),pred_test)))
```


```python
# sampling_results = pd.DataFrame(scores,columns=['Sampling Type','f1','precision','recall','accuracy','auc_roc','auc_pr','confusion_matrix'])
# sampling_results
```


```python
# import xgboost as xgb
# xg_train = xgb.DMatrix(stroke_prepared, label=stroke_labels);
# param = {'max_depth':5, 'eta':0.02, 'silent':1, 'objective':'binary:logistic'}
# num_round = 5000
# early_stopping = 5

# def fpreproc(dtrain, dtest, param):
#     label = dtrain.get_label()
#     ratio = float(np.sum(label == 0)) / np.sum(label == 1)
#     param['scale_pos_weight'] = ratio
#     return (dtrain, dtest, param)

# cv_results = xgb.cv(param, xg_train, num_round, nfold=5,
#        metrics={'auc'}, seed=0, fpreproc=fpreproc, early_stopping_rounds=early_stopping)

# print((cv_results["test-auc-mean"]).tail(1))
```

**Setting up which sampling method to use**

Giving the time constraint, I use the simple and fast under-sampling technique by randomly selecting a subset of data for the smoke classes. This is achieved here by setting the hyperparemeter *over* in the **OverUnderSampling** class defined above, to be False.  Note that by setting it to be True, the SMOTE over-sampling technique is used.


```python
over_samp = OverUnderSampling(over=True)
X_train, y_train = over_samp.transform(X, y)
```

## Voting classifier


```python
log_clf = LogisticRegression(penalty = 'l1')
rnd_clf = RandomForestClassifier()
svm_clf = SVC(probability=True)
tree_clf = DecisionTreeClassifier()
xgb_clf = XGBClassifier()

voting_clf = VotingClassifier(
    estimators=[("lr", log_clf), ("rf", rnd_clf), ("svc", svm_clf), ("dt", tree_clf), ("xb", xgb_clf)],
    voting="soft"
)

voting_clf.fit(X_train, y_train)
scores = []
fpr_s = dict()
tpr_s = dict()
roc_auc = dict()
k = 0
for clf in (log_clf, rnd_clf, tree_clf, svm_clf, xgb_clf, voting_clf):
    clf.fit(X_train, y_train, )
    #x_pred = clf.predict(X_train)
    
    pred_test = clf.predict(X_val.values)
    pred_test_probs = clf.predict_proba(X_val.values)
    probs = clf.predict(X_val.values)
    fpr, tpr, thresholds = roc_curve(y_val.values.ravel(),pred_test)
    p,r,t = precision_recall_curve(y_val.values.ravel(),probs)
    fpr_s[k]=fpr
    tpr_s[k]=tpr
    k +=1
    scores.append((clf.__class__.__name__,
                   f1_score(y_val.values.ravel(),pred_test),
                   precision_score(y_val.values.ravel(),pred_test),
                   recall_score(y_val.values.ravel(),pred_test),
                   accuracy_score(y_val.values.ravel(),pred_test),
                   auc(fpr, tpr),
                   auc(p,r,reorder=True),
                   confusion_matrix(y_val.values.ravel(),pred_test)))    #print(clf.__class__.__name__," training:", precision_score(y_train, x_pred),  recall_score(y_train, x_pred) ,f1_score(y_train, x_pred))
```


```python
sampling_results = pd.DataFrame(scores,columns=['Sampling Type','f1','precision','recall','accuracy','auc_roc','auc_pr','confusion_matrix'])
sampling_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sampling Type</th>
      <th>f1</th>
      <th>precision</th>
      <th>recall</th>
      <th>accuracy</th>
      <th>auc_roc</th>
      <th>auc_pr</th>
      <th>confusion_matrix</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LogisticRegression</td>
      <td>0.102663</td>
      <td>0.054880</td>
      <td>0.794007</td>
      <td>0.741237</td>
      <td>0.767121</td>
      <td>0.407721</td>
      <td>[[10404, 3651], [55, 212]]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RandomForestClassifier</td>
      <td>0.074398</td>
      <td>0.052550</td>
      <td>0.127341</td>
      <td>0.940930</td>
      <td>0.541863</td>
      <td>0.079437</td>
      <td>[[13442, 613], [233, 34]]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DecisionTreeClassifier</td>
      <td>0.053488</td>
      <td>0.038786</td>
      <td>0.086142</td>
      <td>0.943164</td>
      <td>0.522794</td>
      <td>0.052340</td>
      <td>[[13485, 570], [244, 23]]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SVC</td>
      <td>0.100169</td>
      <td>0.053525</td>
      <td>0.779026</td>
      <td>0.739073</td>
      <td>0.758670</td>
      <td>0.399693</td>
      <td>[[10377, 3678], [59, 208]]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>XGBClassifier</td>
      <td>0.109777</td>
      <td>0.060423</td>
      <td>0.599251</td>
      <td>0.818810</td>
      <td>0.711116</td>
      <td>0.314930</td>
      <td>[[11567, 2488], [107, 160]]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>VotingClassifier</td>
      <td>0.112373</td>
      <td>0.066856</td>
      <td>0.352060</td>
      <td>0.896313</td>
      <td>0.629356</td>
      <td>0.196855</td>
      <td>[[12743, 1312], [173, 94]]</td>
    </tr>
  </tbody>
</table>
</div>




```python
sampling_results.to_clipboard
```




    <bound method NDFrame.to_clipboard of             Sampling Type        f1  precision    recall  accuracy   auc_roc  \
    0      LogisticRegression  0.102663   0.054880  0.794007  0.741237  0.767121   
    1  RandomForestClassifier  0.074398   0.052550  0.127341  0.940930  0.541863   
    2  DecisionTreeClassifier  0.053488   0.038786  0.086142  0.943164  0.522794   
    3                     SVC  0.100169   0.053525  0.779026  0.739073  0.758670   
    4           XGBClassifier  0.109777   0.060423  0.599251  0.818810  0.711116   
    5        VotingClassifier  0.112373   0.066856  0.352060  0.896313  0.629356   
    
         auc_pr             confusion_matrix  
    0  0.407721   [[10404, 3651], [55, 212]]  
    1  0.079437    [[13442, 613], [233, 34]]  
    2  0.052340    [[13485, 570], [244, 23]]  
    3  0.399693   [[10377, 3678], [59, 208]]  
    4  0.314930  [[11567, 2488], [107, 160]]  
    5  0.196855   [[12743, 1312], [173, 94]]  >



**XGboost** classifier seems to outperform other classifiers. Let's find some optimal parameters.


```python
gbm_param_grid = {
    'n_estimators': [20, 100],
    'max_depth': range(2, 12),
    'eta': [0.01, 0.02,0.05]
}
# Instantiate the regressor: gbm
gbm = XGBClassifier(n_estimators=10)
# gbm = RandomForestClassifier(class_weight="balanced")
# Perform random search: grid_search
grid_search = RandomizedSearchCV(param_distributions=gbm_param_grid, estimator=gbm, scoring="roc_auc", n_iter=5, cv=4, verbose=1)
# Fit grid_search to the data
grid_search.fit(X_train, y_train)
# Print the best parameters and highest auc
print("Best parameters found: ", grid_search.best_params_)
print("Highest AUC found: ", grid_search.best_score_)
```

    Fitting 4 folds for each of 5 candidates, totalling 20 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done  20 out of  20 | elapsed:  2.2min finished


    Best parameters found:  {'n_estimators': 100, 'max_depth': 11, 'eta': 0.01}
    Highest AUC found:  0.9980659010979366



```python
type(X_val)
```




    pandas.core.frame.DataFrame




```python
final_model = grid_search.best_estimator_
y_prob = final_model.predict_proba(X_val.values)
# len(y_prob)
# help(final_model.predict)
# y_prob = cross_val_predict(final_model, X_val, y_val, cv=3, method="predict_proba")
# y_scores = y_prob[:,1]
fpr, tpr, thresholds = roc_curve(y_val, y_prob[:,1])

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr, tpr)
plt.show()
# precisions, recalls, thresholds = precision_recall_curve(y_val, y_scores)
# def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
#     plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
#     plt.plot(thresholds, recalls[:-1], "g-1", label="Recall")
#     plt.xlabel("Threshold")
#     plt.legend(loc="center left")
#     plt.ylim([0, 1])

# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
# plt.show()
```


![png](output_54_0.png)



```python
precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob[:,1])
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-1", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
```


![png](output_55_0.png)



```python
threshold = 0.02
y_filter = (y_prob[:,1]>threshold)
print(precision_score(y_val, y_filter))
print(recall_score(y_val, y_filter))
```

    0.03625783348254252
    0.9101123595505618


## Feature importances


```python
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
```




    array([0.37547088, 0.02352196, 0.03351074, 0.04378207, 0.06420995,
           0.        , 0.05439086, 0.        , 0.07347387, 0.        ,
           0.06931271, 0.04427072, 0.        , 0.04201558, 0.        ,
           0.03127061, 0.        , 0.07751489, 0.06725512], dtype=float32)



**Extract attribute names**


```python
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.get_feature_names(input_features=low_cardinality_cols))
attributes = numeric_cols + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
```




    [(0.37547088, 'age'),
     (0.077514894, 'smoking_status_never smoked'),
     (0.07347387, 'work_type_Govt_job'),
     (0.069312714, 'work_type_Private'),
     (0.06725512, 'smoking_status_smokes'),
     (0.06420995, 'gender_Male'),
     (0.054390863, 'ever_married_No'),
     (0.044270724, 'work_type_Self-employed'),
     (0.043782067, 'gender_Female'),
     (0.042015582, 'Residence_type_Rural'),
     (0.033510745, 'bmi'),
     (0.03127061, 'smoking_status_formerly smoked'),
     (0.023521956, 'avg_glucose_level'),
     (0.0, 'work_type_children'),
     (0.0, 'work_type_Never_worked'),
     (0.0, 'smoking_status_never smoke'),
     (0.0, 'gender_Other'),
     (0.0, 'ever_married_Yes'),
     (0.0, 'Residence_type_Urban')]




```python
# grid_search.best_estimator_.get_booster().feature_names = attributes
plot_importance(final_model)
plt.show()
attributes
```


![png](output_61_0.png)





    ['age',
     'avg_glucose_level',
     'bmi',
     'gender_Female',
     'gender_Male',
     'gender_Other',
     'ever_married_No',
     'ever_married_Yes',
     'work_type_Govt_job',
     'work_type_Never_worked',
     'work_type_Private',
     'work_type_Self-employed',
     'work_type_children',
     'Residence_type_Rural',
     'Residence_type_Urban',
     'smoking_status_formerly smoked',
     'smoking_status_never smoke',
     'smoking_status_never smoked',
     'smoking_status_smokes']



The factors influencing stroke are: **age, bmi, avg_glucose_level, smoking_status, ever_married**. Howerver, a further analysis with only the first three factors should be investigated.


```python
# param_grid = [
#     {"n_estimators": [3, 10, 30], "max_features": [2,4,6,8]},
#     {"bootstrap": [False], "n_estimators": [3,10], "max_features": [2, 3, 4]},
# ]

# xgb_clf = XGBClassifier()
# grid_search = GridSearchCV(xgb_clf, param_grid, cv=5, scoring="")
# y_train_pred = cross_val_predict(xgb_clf, X_train, y_train, cv=3)
# y_test_pred = cross_val_predict(xgb_clf, X_test, y_test, cv=3)
# print(precision_score(y_train_pred, y_train), recall_score(y_train_pred, y_train) ,f1_score(y_train, x_pred))
# print(precision_score(y_test_pred, y_val), recall_score(y_test_pred, y_val), ,f1_score(y_test_pred, y_val) )

# accuracy_score(y_train_pred, y_train)
# confusion_matrix(y_train, y_train_pred)
```


```python
fpr, tpr, thresholds = roc_curve(y_val, y_scores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], "k--")
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    
plot_roc_curve(fpr, tpr)

y_probas_forest = cross_val_predict(rnd_clf, X_val, y_val, cv=3, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_val, y_scores_forest)

y_probas_xgb = cross_val_predict(xgb_clf, X_val, y_val, cv=3, method="predict_proba")
y_scores_xgb = y_probas_xgb[:, 1]
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_val, y_scores_xgb)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plot_roc_curve(fpr_xgb, tpr_xgb, "XGBoost")
plt.legend(loc="lower right")
plt.show()
```
