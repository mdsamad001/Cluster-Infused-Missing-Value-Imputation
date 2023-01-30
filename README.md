# Cluster Infused Missing Value Imputation

## Usage
Please run below commands to prepare your enviroment:

`pip install numpy==1.19.5`
`pip install pandas==1.3.3`
`pip install scikit-learn=0.24.2`
`pip install -e micegradient`

You can run the mice.ipynb jupyter notebook file and check the results for the dummy_dataset.
When using your own dataset, make sure to remove all non-numerical data columns before proceeding.


## Citation
You can find more details in our publication: [Missing value estimation using clustering and deep learning within multiple imputation framework](https://www.sciencedirect.com/science/article/abs/pii/S0950705122004695)

You can copy below bibtex for citation:
```
@article{SAMAD2022108968,
title = {Missing value estimation using clustering and deep learning within multiple imputation framework},
journal = {Knowledge-Based Systems},
volume = {249},
pages = {108968},
year = {2022},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2022.108968},
url = {https://www.sciencedirect.com/science/article/pii/S0950705122004695},
author = {Manar D. Samad and Sakib Abrar and Norou Diawara},
keywords = {Missing value imputation, Ensemble learning, Deep learning, Multiple imputation, MICE, Clustering},
abstract = {Missing values in tabular data restrict the use and performance of machine learning, requiring the imputation of missing values. Arguably the most popular imputation algorithm is multiple imputation by chained equations (MICE), which estimates missing values from linear conditioning on observed values. This paper proposes methods to improve both the imputation accuracy of MICE and the classification accuracy of imputed data by replacing MICE’s linear regressors with ensemble learning and deep neural networks (DNN). The imputation accuracy is further improved by characterizing individual samples with cluster labels (CISCL) obtained from the training data. Our extensive analyses of six tabular data sets with up to 80% missing values and three missing types (missing completely at random, missing at random, missing not at random) reveal that ensemble or deep learning within MICE is superior to the baseline MICE (b-MICE), both of which are consistently outperformed by CISCL. Results show that CISCL + b-MICE outperforms b-MICE for all percentages and types of missing values. In most experimental cases, our proposed DNN-based MICE and gradient boosting MICE plus CISCL (GB-MICE-CISCL) outperform seven state-of-the-art imputation algorithms. The classification accuracy of GB-MICE-imputed data is further improved by our proposed GB-MICE-CISCL imputation method across all percentages of missing values. Results also reveal a shortcoming of the MICE framework at high percentages of missing values (>50%) and when the missing type is not random. This paper provides a generalized approach to identifying the best imputation model for a tabular data set based on the percentage and type of missing values.}
}
```
