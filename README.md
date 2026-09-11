##### 5 sample records:

      hours_studied  attendance  ...  sleep_duration  passed

891 22.76 90.82 ... 6.85 1
1338 23.75 98.14 ... 9.94 1
1694 24.13 82.20 ... 7.25 1
805 22.47 72.49 ... 5.06 0
1684 22.09 65.45 ... 5.81 0

[5 rows x 6 columns]

##### Data shape:

(2500, 6)

##### Data size:

15000

##### Data column names:

Index(['hours_studied', 'attendance', 'homework_completion', 'previous_scores',
'sleep_duration', 'passed'],
dtype='object')

##### X sample:

      hours_studied  attendance  ...  previous_scores  sleep_duration

2177 14.29 66.27 ... 48.00 8.39
2036 10.69 85.38 ... 46.16 6.38
320 1.14 97.04 ... 92.34 8.86
479 9.97 85.12 ... 78.89 6.56
2398 6.43 61.80 ... 61.64 5.37

[5 rows x 5 columns]

##### X shape:

(2500, 5)

##### y sample:

2438 1
2280 1
24 1
2455 0
1861 0
Name: passed, dtype: int64

##### y shape:

(2500,)

##### X_train shape:

(1875, 5)

##### X_test shape:

(625, 5)

##### y_train shape:

(1875,)

##### y_test shape:

(625,)

##### Accuracy logistic regression:

0.8304

##### F1 logistic regression:

0.8215488215488216

##### MAE logistic regression:

0.1696

##### RMSE logistic regression:

0.41182520563948

##### Confusion matrix logistic regression:

[[275  45]
 [ 61 244]]

##### X_train shape:

(1875, 5)

##### X_test shape:

(625, 5)

##### y_train shape:

(1875,)

##### y_test shape:

(625,)

##### Accuracy decision tree:

0.7344

##### F1 decision tree:

0.7251655629139073

##### MAE decision tree:

0.2656

##### RMSE decision tree:

0.5153639490690051

##### Confusion matrix decision tree:

[[240  80]
 [ 86 219]]
