# Post-Mendelian genetic model in COVID-19

* featurize.py: Generation of the boolean representation from variants annoted with VEP

* analysis.py: Feature extraction by logistic regression with LASSO regularization

For computing the adjusted phenotype, a CSV file with the following format is required:
* One column named sample with the unique ID of the sample
* One column named age with age in years
* One column named gender with the sex coded as 0/1, with 1 for females
* One column named grading coded as:
  * 0 = not hospitalized
  * 1 = hospitalized without respiratory support
  * 2 = hospitalized with O2 supplementation
  * 3 = hospitalized with CPAP-biPAP
  * 4  = hospitalized intubated
Only PCR-positive individuals should be included.

