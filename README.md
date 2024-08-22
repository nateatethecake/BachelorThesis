# BachelorThesis
This Thesis is about assessing TabPFN's vulnerability towards membership inference attacks.

## Code Resources
1) As reference for membership inference attack I used the approach from the following paper 'Membership Inference from First Principles:
https://arxiv.org/abs/2112.03570
The paper also includes a coding example:
https://github.com/tensorflow/privacy/tree/master/research/mi_lira_2021

2) To create the Synthetic Dataset by Shokri
https://arxiv.org/pdf/1610.05820

3) The formal use of TabPFN has been introduced in this paper:
https://arxiv.org/abs/2207.01848
Additionally they have provided coding examples.
https://github.com/automl/TabPFN


## Datasets:
The datasets have been retrived from the OpenML database.
In this project the focus lays on supervised learning e.g. classification. 
I mainly focused on datasets that contain medical, sensitive information.

For now, i have settled for 5 datasets to apply a training with TabPFN:
1) Hepatitis (https://www.openml.org/search?type=data&status=active&id=55): 55
2) Diabetes (https://www.openml.org/search?type=data&sort=runs&id=37&status=active): 37
3) Breast-W (https://www.openml.org/search?type=data&status=active&id=15): 15
4) Heart-statlog (https://www.openml.org/search?type=data&status=active&id=53): 53
5) Bloodtransfusion (https://www.openml.org/search?type=data&status=active&id=1464): 1464


