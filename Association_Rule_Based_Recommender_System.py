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
# ServiceId     : Anonymized services related to each category
# CategoryId    : Anonymized categories (Example: Cleaning, transportation, renovation).
# CreateDate    : Date the service was purchased.

# Project Tasks
# Task 1: Data Prepration
# Task 2: Generate Association Rules and Provide Recommendations



import pandas as pd
pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules

# -----------------------------------------------------------------------------
# Task 1: Data Prepration
# -----------------------------------------------------------------------------

# Loading the armut_data.csv file.

df_ = pd.read_csv(r"Association_Rule_Based_Recommender_System\armut_data.csv")
df = df_.copy()
df.head()

df['Hizmet'] = df['ServiceId'].astype(str)+"_"+df['CategoryId'].astype(str)


# Each ServiceId represents a distinct service for every CategoryID.
# Create a new variable by combining ServiceID and CategoryID with "_" which will represent these services.


#  output for this step:
# -----------------------------------------------------------------------------
# Sample Data:
# UserId | ServiceId | CategoryId | CreateDate       | Hizmet
# 25446  | 4         | 5          | 6.08.2017 16:11  | 4_5
# 22948  | 48        | 5          | 6.08.2017 16:12  | 48_5
# 10618  | 0         | 8          | 6.08.2017 16:13  | 0_8
# 7256   | 9         | 4          | 6.08.2017 16:14  | 9_4
# 25446  | 48        | 5          | 6.08.2017 16:16  | 48_5
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# The dataset is comprised of the date and time the services were received.
# There's no specific cart definition (like an invoice). For implementing
# Association Rule Learning, we need to define a 'cart'. Here, the cart is
# defined as the services availed by each customer monthly.


df['CreateDate'] = pd.to_datetime(df['CreateDate'])

df["New_Date"] = df["CreateDate"].dt.strftime("%Y-%m")

df['SepetID'] = df['UserId'].astype(str)+"_"+df['New_Date']


# -----------------------------------------------------------------------------
# Task 2: Generate Association Rules and Provide Recommendations
# -----------------------------------------------------------------------------

# Convert the data into an invoice-product matrix
invoice_product_df = df.groupby(['SepetID', 'Hizmet'])['Hizmet'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
invoice_product_df.head()

# Generating association rules.
frequent_itemsets = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()

# Using the arl_recommender function to suggest a service for a user who last availed the service "2_0".
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    # Sort the rules by lift in descending order. (To capture the most compatible product first)
    # Sorting by confidence is also an option depending on discretion.
    recommendation_list = [] # Create an empty list for recommended products.
    # antecedents: X
    # It fetches the items as frozensets. Combines index and product.
    # i: index
    # product: X i.e., the service asking for recommendation
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product): # Loop through the services (product):
            if j == product_id: # If the product asking for recommendation is found:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
                # Store the consequents(Y) value of the current index to the recommendation list.

    # To prevent repetition in the recommendation list:
    # For instance, the same product might reappear in multiple combinations (like pairs or triplets);
    # Leveraging the unique nature of the dictionary structure.
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count] # Return recommendations up to the desired number (rec_count).

arl_recommender(rules,"2_0", 4)

# Out[]: ['22_0', '38_4', '2_0']