# -----------------------------------------------------------------------------
# -                             Business Problem                              -
# -----------------------------------------------------------------------------

# Armut, Turkey's largest online service platform, connects service providers with those who seek services.
# It offers quick and easy access to services such as cleaning, renovation, and transportation with just a few taps on a computer or smartphone.
# Using a dataset that contains information about the users and the services and
# categories they have availed, an Association Rule Learning-based recommendation system is desired to be built.

# -----------------------------------------------------------------------------
# -                               Dataset Story                               -
# -----------------------------------------------------------------------------

# The dataset consists of services availed by customers and the categories of these services.
# It also includes the date and time of each service. The dataset has 4 variables with 162,523 observations and is 5 MB in size.

# -----------------------------------------------------------------------------
# -                               VARIABLES                                   -
# -----------------------------------------------------------------------------

# UserId        : Customer number
# ServiceId     : Anonymized services related to each category (Example: Under the cleaning category, there's a service for couch cleaning). A ServiceId can appear under different categories and may represent different services in each category. (Example: A service with CategoryId 7 and ServiceId 4 is for radiator cleaning, while a service with CategoryId 2 and ServiceId 4 is for furniture assembly).
# CategoryId    : Anonymized categories (Example: Cleaning, transportation, renovation).
# CreateDate    : Date the service was purchased.

# Project Tasks
# Task 1: Data Prepration
# Task 2: Generate Association Rules and Provide Recommendations



import pandas as pd
pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules

# -----------------------------------------------------------------------------
# Task 1: Data Prepration                                                     -
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Step 1: Load the armut_data.csv file.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Step 2: Each ServiceId represents a distinct service for every CategoryID.
# Create a new variable by combining ServiceID and CategoryID with "_",
# which will represent these services.
# Expected output for this step:
# -----------------------------------------------------------------------------

# (Here you can write the code to handle this step)

# -----------------------------------------------------------------------------
# Step 3: The dataset is comprised of the date and time the services were received.
# There's no specific cart definition (like an invoice). For implementing
# Association Rule Learning, we need to define a 'cart'. Here, the cart is
# defined as the services availed by each customer monthly.
# Example: The customer with ID 7256 has services 9_4 and 46_4 in the 8th
# month of 2017 representing one cart; whereas services 9_4 and 38_4 in the
# 10th month of 2017 represent another cart. These carts need unique identifiers.
# Start by creating a new date variable only containing year and month.
# Then concatenate UserID with the new date variable using "_" to derive
# a new variable named 'ID'.
# Expected output:
# -----------------------------------------------------------------------------

# (Here you can write the code to handle this step)

# -----------------------------------------------------------------------------
# Task 1: Data Preparation
# -----------------------------------------------------------------------------
# Sample Data:
# UserId | ServiceId | CategoryId | CreateDate       | Hizmet
# 25446  | 4         | 5          | 6.08.2017 16:11  | 4_5
# 22948  | 48        | 5          | 6.08.2017 16:12  | 48_5
# 10618  | 0         | 8          | 6.08.2017 16:13  | 0_8
# 7256   | 9         | 4          | 6.08.2017 16:14  | 9_4
# 25446  | 48        | 5          | 6.08.2017 16:16  | 48_5
# -----------------------------------------------------------------------------

# (Here you can write the code to handle data preparation)







# Adım 1: armut_data.csv dosyasınız okutunuz.
df_ = pd.read_csv(r"Association_Rule_Based_Recommender_System\armut_data.csv")
df = df_.copy()
df.head()

def analyze_missing_values(df):
    na_cols = df.columns[df.isna().any()].tolist()
    total_missing = df[na_cols].isna().sum().sort_values(ascending=False)
    percentage_missing = ((df[na_cols].isna().sum() / df.shape[0]) * 100).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Count': total_missing, 'Percentage (%)': np.round(percentage_missing, 2)})
    return missing_data

# to get an initial understanding of the data's structure, its content, and if there are any missing values that need to be addressed.
def sum_df(dataframe, head=6):
    print("~~~~~~~~~~|-HEAD-|~~~~~~~~~~ ")
    print(dataframe.head(head))
    print("~~~~~~~~~~|-TAIL-|~~~~~~~~~~ ")
    print(dataframe.tail(head))
    print("~~~~~~~~~~|-TYPES-|~~~~~~~~~~ ")
    print(dataframe.dtypes)
    print("~~~~~~~~~~|-SHAPE-|~~~~~~~~~~ ")
    print(dataframe.shape)
    print("~~~~~~~~~~|-NUMBER OF UNIQUE-|~~~~~~~~~~ ")
    print(dataframe.nunique())
    print("~~~~~~~~~~|-NA-|~~~~~~~~~~ ")
    print(dataframe.isnull().sum())
    print("~~~~~~~~~~|-QUANTILES-|~~~~~~~~~~ ")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("~~~~~~~~~~|-NUMERIC COLUMNS-|~~~~~~~~~~ ")
    print([i for i in dataframe.columns if dataframe[i].dtype != "O"])
    print("~~~~~~~~~~|-MISSING VALUE ANALYSIS-|~~~~~~~~~~ ")
    print(analyze_missing_values(dataframe))

sum_df(df)

# ###############################################################################################################
# Task 2: Generate Association Rules and Provide Recommendations
# ###############################################################################################################