## AutoML for Predicting Telco Churn
### Overview

| First Header  | Second Header |
| ------------- | ------------- |
| TIME REQUIRED | a few minutes |
| COMPLEXITY  | simple |
| TAGS  | automl, optimizer, binary, preparation |
| SOURCE | https://www.kaggle.com/blastchar/telco-customer-churn |


### Files

This example contains two self-contained notebooks that take you from the raw dataset to experiment results. 

- `telco_churn_prepare.ipynb`
- `telco_churn_experiment.ipynb`

In addition, the contents of `telco_churn_prepare.ipynb` are summarized as a function in `telco_churn.py`, which is loaded in `telco_churn_experiment.ipynb` for convenience. Finally, an experiment log with +800 permutation results is included in `telco_churn_for_sensitivity.csv`.


### Result

This example is intended for an asset for developing optimizers to Talos, and for this purpose a starting point of f1score=0.62 is provided within the included `telco_churn_for_sensitivity.csv` experiment log.
