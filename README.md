# Lenuse Health
Customer segmentation challenge for Analysts

Please complete the following data processing challenge by <INSERT DATE HERE>.
Description

Use the data provided in the file customer_data_sample.csv and, through the use of visualizations and/or statistics answer the question:

**"What are the most important factors for predicting whether a customer has converted or not?"**

Converted customer is represented in the data in the field "converted", and the nature of what this conversion means is (intentionally) unknown in the context of the challenge.

Fields

| field | explanation |
|---|---|
| customer_id | Numeric id for a customer
| converted | Whether a customer converted to the product (1) or not (0)
| customer_segment | Numeric id of a customer segment the customer belongs to
| gender | Customer gender
| age | Customer age
| related_customers | Numeric - number of people who are related to the customer
| family_size | Numeric - size of family members
| initial_fee_level | Initial services fee level the customer is enrolled to
| credit_account_id | Identifier (hash) for the customer credit account. If customer has none, they are shown as "9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0"
| branch | Which branch the customer mainly is associated with |

Submission requirements

Submit your work as a git repository (preferred way):

Via git (github or bitbucket):  

Submit your answer as a version controlled (git) repository (repo) in github or bitbucket. Make sure your repo is public and submit a link to it via email.

Suggested tools / approaches

- Use summary statistics, visualization or other analytical means to explain your argumentation - it's important that you coherently explain, why you deem certain factors important and why some might be considered more important than others
- You can for example use ipython (jupyter) notebooks, BI visualization tools (Tableau, Power BI, Excel) or such
- Remember to include your full answer and used visualizations (code and pdfs) in your submission

Reach out to adithya@lenus.io if you have any questions regarding the case brief or dataset.

# Solution

The solution is nicely explained in the jupyter notebook `Solution.ipynb`. However, for a fast overlook of the work, one can run the 3 modules in the repository, `Cleaning_data.py`, `visualizing.py` and `processing.py`.

To make the debugging as simple as possible, remember to check the list of the needed packages:

* [numpy](https://numpy.org/)

* [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html)

* [matplotlib](https://matplotlib.org/)

* [scikit-learn](https://scikit-learn.org/stable/index.html)

* [xgboost](https://xgboost.ai/)

For installing all the dependencies:

pip install numpy pandas matplotlib scikit-learn xgboost

Also, the version of Python that was use is python 3.9, although any version of python3 higher than 3.6 should work properly.

To run all the modules:

python3 Cleaning_data.py
python3 visualizing.py
python3 processing.py
