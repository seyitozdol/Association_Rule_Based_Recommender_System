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


# Task 1: Data Prepration
## Step 1: Load the armut_data.csv file.

df_ = pd.read_csv(r"Association_Rule_Based_Recommender_System\armut_data.csv")
df = df_.copy()
df.head()

df['Hizmet'] = df['ServiceId'].astype(str)+"_"+df['CategoryId'].astype(str)


## Step 2: Each ServiceId represents a distinct service for every CategoryID.
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
# Step 3: The dataset is comprised of the date and time the services were received.
# There's no specific cart definition (like an invoice). For implementing
# Association Rule Learning, we need to define a 'cart'. Here, the cart is
# defined as the services availed by each customer monthly.

df.dtypes

df['CreateDate'] = pd.to_datetime(df['CreateDate'])

df["New_Date"] = df["CreateDate"].dt.strftime("%Y-%m")

df['SepetID'] = df['UserId'].astype(str)+"_"+df['New_Date']


# ###############################################################################################################
# Task 2: Generate Association Rules and Provide Recommendations
# ###############################################################################################################

invoice_product_df = df.groupby(['SepetID', 'Hizmet'])['Hizmet'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
invoice_product_df.head()

frequent_itemsets = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()


#Adım 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    # kuralları lifte göre büyükten kücüğe sıralar. (en uyumlu ilk ürünü yakalayabilmek için)
    # confidence'e göre de sıralanabilir insiyatife baglıdır.
    recommendation_list = [] # tavsiye edilecek ürünler için bos bir liste olusturuyoruz.
    # antecedents: X
    #items denildigi için frozenset olarak getirir. index ve hizmeti birleştirir.
    # i: index
    # product: X yani öneri isteyen hizmet
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product): # hizmetlerde(product) gez:
            if j == product_id:# eger tavsiye istenen ürün yakalanırsa:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
                # index bilgisini i ile tutuyordun bu index bilgisindeki consequents(Y) değerini recommendation_list'e ekle.

    # tavsiye listesinde tekrarlamayı önlemek için:
    # mesela 2'li 3'lü kombinasyonlarda aynı ürün tekrar düşmüş olabilir listeye gibi;
    # sözlük yapısının unique özelliginden yararlanıyoruz.
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count] # :rec_count istenen sayıya kadar tavsiye ürün getir.



arl_recommender(rules,"2_0", 4)