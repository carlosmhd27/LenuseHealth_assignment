from os.path import join, isfile, isdir
from pandas  import read_excel, factorize, unique
from numpy   import nan

def cleaning_data(name, save_csv = False, print_info = False, output_name = None, dropnan = True):
    '''
    Function to inspect and clean the data in the Excel file
    '''

    assert isfile(name), "The dataset does not exit"
    ## Open the data set as an Excel, it is important to notice that I already separate
    ## the columns with excell, but if one has to it by itself, you can use the function
    ## data[data.keys()[0]].str.split(",",expand=True)
    data = read_excel(name)

    # Some simple inspection to know the data
    if print_info:
        total_users = data.shape[0]
        print(data.head())
        print(f"This dataset has {data.shape[1]} different parameters with {total_users} entries")
        print(f"The columns with nan values are: \n {data.isna().any()}")
    ## Erase all the non important parameters, in this case only the costumer_id
    ## as it is an individual identifier
    data.pop('customer_id')

    for uniq in data["customer_segment"].unique():
        repeated = len(data['customer_segment'][data['customer_segment'] == uniq].index)
        print(f"The customer_segment {uniq} englobes {repeated} / {total_users} costumers")
    ## Not all the people filled the age value, which means that we might want to give a value to that:
    if print_info:
        nan_age = data["age"].isna().sum()
        print(f"Number of users with no age given {nan_age} / {total_users}")
        ## They are not so many and they might be important, erasing the age with a random value, e.g. 100
        ## might generate some biased towards higher values or erase the importance of higher values,
        ## as they are not that many, I will pop them out, the same happens with the branch column,
        ## where are 2 nan values
        if dropnan:
            data = data.dropna();
            total_users = data.shape[0]
            print(f"The number of rows has drop to {total_users}")

    ## Change the credit account into boolean variables, 1 if they gave a credit account
    ## 0 if they have none. The credit_account_id itself it not important, but knowing
    ## if they give this information might be
    data['credit_account_id'] = data['credit_account_id'].where(data['credit_account_id'] == "9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0", "a")
    data['credit_account_id'],_ = factorize(data['credit_account_id'], sort = True)
    if print_info:
        print(f"there is {len(data['credit_account_id'][data['credit_account_id'] == 1].index)} / {total_users} entris with credit_account_id")
        print(f"there is {len(data['credit_account_id'][data['credit_account_id'] == 0].index)} / {total_users} entris with no credit_account_id")


    ## Do the same with the gender, 0 for female and 1 for male
    data['gender'], genders = factorize(data['gender'], sort = True)
    if print_info:
        print(f"there is {len(data['gender'][data['gender'] == 0].index)} / {total_users} {genders[0]} users")
        print(f"there is {len(data['gender'][data['gender'] == 1].index)} / {total_users} {genders[1]} users")


    ## Let's look at the branches
    data['branch'], branches = factorize(data['branch'], sort = True)
    branches = list(branches)

    if print_info:
        print(f"This dataset has {len(branches)} branches, with the names {branches}")

    branches_dic = {}
    for branch in branches:
        branches_dic[branch] = len(data['branch'][data['branch'] == branches.index(branch)].index)

        if print_info:
            print(f"The branch {branch} has {branches_dic[branch]} / {total_users} entries.")
    data['branch'] = data['branch'].astype(int)
    if save_csv:
        if output_name == None:
            output_name = "Cleaned" + name
        data.to_csv(output_name, index = False)

    return data

if (__name__ == "__main__"):

    path_to_dataset = "./"
    dataset_name = join(path_to_dataset, "Data_Scientist_-_Case_Dataset_splitted.xlsx")
    output_name  = join(path_to_dataset, "Cleaned_datashet_Carlos.csv")

    data = cleaning_data(dataset_name,
                        save_csv = True,
                        print_info = True,
                        output_name = output_name)