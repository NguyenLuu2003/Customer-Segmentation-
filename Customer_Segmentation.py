#!/usr/bin/env python
# coding: utf-8

# # Project
# 
# ## CUSTOMER SEGMENTATION WITH RFM MODEL AND K-MEANS CLUSTERING ALGORITHM BY USING PYSPARK IN PYTHON
# 
# The primary objective of this project is to facilitate the development of customer-centric strategies through the integration of Customer Relationship Management (CRM) with Analytics. The method used in this project are RFM (Recency, Frequency, Monetary) Analysis and K-Means Machine Learning model to classify so that businesses can gain deeper insights into customer behavior and preferences. Since the dataset is big, we are going to use Pyspark to handle and analyze the data in order to achieve best results.
# 
# 
# 

# # I. Project overview
# 

# ## 1. Research objectives
# 

#    The purpose of the project is to segment customers in business – an important element indispensable in the business and marketing strategy of any business. It is especially important in the context of an increasingly competitive market and the diversity of customer needs. The goal of this segmentation is to create a deep understanding of the needs, wants, and behaviors of each customer group so that they can best be served.
# 
#    Customer segmentation is the process by which a business divides its customers into smaller groups based on certain similarities: interests, wants, needs, personality, age, and many other factors. The key to effective segmentation is to divide customers into groups and make predictions about their value to the business. Then, target each customer group with different strategies to extract maximum value from customers with high and low profitability.
# 
#    Effective customer segmentation will bring a lot of benefits to businesses. To begin with, it helps to optimize the marketing strategy by categorizing customers into groups, businesses can create separate marketing strategies for each segment. This helps to focus resources and resources on the most effective advertising, marketing, and promotion campaigns. Moreover, it creates personalized experiences because understanding each segment allows businesses to create products, services, and experiences based on their specific needs and improve quality for the better. Customers will feel more satisfied and interested in the product when receiving personalization. In particular, it gives businesses the opportunity to build long-term and loyal relationships with customers. When customers feel they are understood and responsive, they will tend to come back and support the business in the future. As a result, businesses will grow revenue, and profit and improve their brand reputation much more.
# 

# ## 2. Description of project
# 

# In the project, we use a transnational data set that contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail. Based on the data set, we aim to perform RFM (Recency, Frequency, and Monetary Value) analysis on the data to be able to segment the customers based on their purchasing patterns and behavior. Data scientists usually build unsupervised machine learning algorithms such as K-Means clustering or hierarchical clustering to perform customer segmentation. These models are great at identifying similar patterns between user groups that often go unnoticed by the human eye. So, using K-Means can help us segment our customers into many groups, then we can have suitable services for them.
# 
# The reason we chose this method is because many previous studies have used it and achieved good results. The article by Edwin Omol, Dorcas Onyango, Lucy Mburu (2024) explores the application of the K-means clustering algorithm for customer segmentation in grocery stores within the unique context of Kenya. The application of K-means clustering for customer segmentation in grocery stores in Kenya yielded valuable insights into the distinct customer segments and their preferences. The results indicate that there is not a significant difference between male and female customers, it is also observed that customers in the age range of 20-40 tend to spend more compared to other age group.The findings of this study have significant implications for grocery stores in Kenya, enabling them to enhance their marketing strategies and improve overall customer experiences. The application of K-means clustering facilitated the creation of targeted approaches tailored to the preferences of different customer segments. By understanding the distinct needs and behaviors of each cluster, store owners and marketers can develop more effective marketing campaigns and personalized promotions. Furthermore, the study of MA Syakur, BK Khotimah, EMS Rochman, BD Satoto (2018) had also suggested using K-Means Clustering Method and Elbow Method for identification of the best customer profile. The results obtained from the process in determining the best number of clusters with elbow and K-Means methods that the determination of the best number of clusters can produce the same number of clusters K on the amount of different data. The result of determining the best number of clusters with elbow method will be the default for characteristic process based on case study.
# 
# 

# # II. Project methods

# ## 1. RFM model

# ### 1.1. Definition of RFM model

# RFM analysis is a marketing technique used to quantitatively rank and group customers based on the recency, frequency and monetary total of their recent transactions to identify the best customers and perform targeted marketing campaigns. The system assigns each customer numerical scores based on these factors to provide an objective analysis. RFM analysis is based on the marketing adage that "80% of your business comes from 20% of your customers."
# 
# RFM analysis ranks each customer on the following factors:
# - Recency: How recent was the customer's last purchase? Customers who recently made a purchase will still have the product on their mind and are more likely to purchase or use the product again. Businesses often measure recency in days. But, depending on the product, they may measure it in years, weeks or even hours.
# - Frequency: How often did this customer make a purchase in a given period? Customers who purchased once are often more likely to purchase again. Additionally, first time customers may be good targets for follow-up advertising to convert them into more frequent customers.
# - Monetary: How much money did the customer spend in a given period? Customers who spend a lot of money are more likely to spend money in the future and have a high value to a business.
# 

# ### 1.2. The importance of RFM analysis
# 

# Customer segmentation based on RFM can help you evaluate the customer journey based on many factors, from many perspectives. RFM analysis can help answer a number of questions such as:
# 
# - Who are the customers who spend the most and are most loyal?
# - Which customers are most likely to leave the company?
# - Who are the new customers and how can they increase the likelihood of recurring purchases?
# - Which customers could potentially convert to higher spending customers?
# - Which customers have left the brand and are least likely to return?
# - Which customers need to be cared for and retained before becoming churn customers?
# - How do different customer segments require different care, incentives and marketing activities?
# 
# RFM analysis helps you build different messages, campaigns and incentive programs for each customer group, based on their consumer behavior and relationship with the company. Each of these RFM metrics has been proven to be effective in predicting future customer behavior and increasing revenue. Customers who have made purchases in the recent past are more likely to do so in the near future. Based on this, marketing activities can increase response rates, increase customer retention (retention rate), customer satisfaction and customer lifetime value (CLTV).
# 
# In short, RFM analysis can help you achieve several benefits. First, it can help increase email marketing effectiveness and improve customer lifetime value. Secondly, it can help with customer segmentation for new product introductions. Loyal customer groups are highly likely to receive, use and respond positively to the company's new products. Thirdly, RFM would help increase loyalty and increase interaction with customers and reduce the rate of customers leaving your business. Finally, the methods would help optimize marketing costs and improve the effectiveness of remarketing/retargeting campaigns.
# 

# ### 1.3. How RFM analysis work?
# 

# RFM analysis scores customers on each of the three main factors. Generally, a score from 1 to 5 is given, with 5 being the highest. Various implementations of an RFM analysis system may use slightly different values or scaling, however.
# 
# The collection of three values for each customer is called an RFM cell. In a simple system, organizations average these values together, then sort customers from highest to lowest to find the most valuable customers. Some businesses, instead of simply averaging the three values, weigh the values differently.
# 
# For example, a car dealership may recognize that an average customer is highly unlikely to buy several new cars in a timeframe of just a few years. But a customer who does buy several cars -- a high-frequency customer -- should be highly sought after. So, the dealership may choose to weigh the value of the frequency score accordingly.
# 

# ## 2. K-Means clustering algorithm
# 

# ### 2.1. K-means clustering
# 

# The K-means method was stated by MacQueen (1967) and Lloyd (1982). In the K-means clustering algorithm, we do not know the label of each data point. The goal is to divide data into different clusters so that data in the same cluster have similar properties. Similar objects are grouped in the same cluster using iterative refinement with the goal of minimizing the loss function computed by the sum of squared errors between each cluster centroid and the elements in the cluster.
# 
#  K-means clustering divides elements into clusters, each element is assigned to the cluster with the center (average of the elements) closest to it. In each refinement, the goal is to minimize the sum of squared distances:
# 

# The stop condition for the algorithm: When the centers do not change in 2 consecutive iterations. However, achieving a perfect result is very difficult and very time-consuming, so people will usually stop the algorithm when an approximate and acceptable result is achieved.
# 

# ### 2.2. Algorithm implementation steps

# Step 1: Find k initial cluster centers (can be chosen by randomly selecting k elements).
# 
# Step 2: Calculate the distance of each element to the centers. The element will be assigned to the cluster with the center closest to it.
# 
# Step 3: Recalculate the k cluster centers by averaging all the elements assigned to each cluster.
# 
# Step 4:  Repeat steps 2 and 3 until the cluster centers no longer change.
# 
# When using K-means, there are two important factors to consider: calculating the distance between points to the centroid and the number of centroids.
# 

# ### 2.3. Calculating the distance of each point to the centroids
# 

# a. Determine the label for each point based on the Ci data on the distance to each cluster center:
# 

# b. Recalculate the center for each cluster according to the average of all data points in a cluster:
# 
# 

# ### 2.4. Elbow method in deciding K
# 

# The Elbow method is an exclusive method in deciding the numbers of centroids should be made when using the k-means method. In the k-mean method, it’s important to decide the number of k centroids before starting the method. The Elbow method is a way to help us choose the appropriate number of clusters based on the visualization graph by looking at the decline of the deformation function and selecting the elbow point.
# 
# The elbow point is the point where the decay rate of the deformation function will change the most. That is, from this position on, increasing the number of clusters will not help the deformation function decrease significantly. If the algorithm divides according to the number of clusters at this position, it will achieve the most general clustering properties without encountering overfitting phenomena.
# 

# In which |Si | is the number of observations in cluster i and d (x,y) is the distance between observations x and y. The "optimal" number of clusters is the k value such that when we increase the number of clusters, the explanatory efficiency of clustering does not improve much.
# 
# 

# # II. Data Analysis

# ## 1. Overview of the dataset

# This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.
# 

# Variables Description:
# This section provides a detailed description of the variables included in this E-commerce dataset:
# 
# - InvoiceNo: Each transaction is uniquely identified by its invoice number. Invoice numbers prefixed with "C" denote refund transactions.
# - StockCode: An exclusive code assigned to each item in the inventory.
# - Description: The descriptive name of the item being purchased.
# - Quantity: The quantity of items included in the transaction.
# - InvoiceDate: Date and time when the purchase transaction occurred.
# - UnitPrice: The price of a single item, denominated in Sterling currency.
# - CustomerID: A unique identifier assigned to each customer.
# - Country: The country of residence for the customer.

# ## 2. Creating a SparkSession & DataFrame:

# A SparkSession is an entry point into all functionality in Spark, and is required if you want to build a dataframe in PySpark.

# In[1]:


pip install pyspark


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("PySpark").getOrCreate()
spark


# In[3]:


from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window


# In[4]:


def printdf(df, l=5):
    return df.limit(l).toPandas()

def nullcount(df):
    return {col: df.filter(df[col].isNull()).count() for col in df.columns}

def shape(df):
    print((df.count(), len(df.columns)))


# In[8]:


data = spark.read.csv("E-commerce.csv",
                      inferSchema=True,
                      header=True)
print('Dataframe dimensions:', (data.count(), len(data.columns)))
printdf(data)


# ## 3. Data Cleaning & Exploratory Data Analysis

# In[9]:


data.toPandas().describe()


# In[10]:


nullcount(data)


# In[11]:


#Count number of customerID in each country, then calculate the percentage of each country with the total
rtl_data = data["Country", "CustomerID"].distinct()\
    .groupBy("Country")\
    .agg(F.count("CustomerID").alias("Count"))\
    .withColumn('Total', F.sum('Count').over(Window.partitionBy()))\
    .withColumn('%', (F.col('Count')/F.col('Total'))*100)\
    .sort("Count", ascending=False)

printdf(rtl_data)


# We will consider only UK customers as they account for the majority of the data (90%)

# In[12]:


# Consider only customers from UK
rtl_data = data.filter(F.col("Country") == "United Kingdom")

# Filter out null customer ids
rtl_data = rtl_data.filter(F.col("CustomerID").isNotNull())

rtl_data.toPandas().shape

printdf(rtl_data)


# In[13]:


nullcount(rtl_data)


# ## 4. Data Pre-processing & Standardization

# After analyzing the dataset and having a better understanding of each data point, the next step is preparing the data to feed into the machine learning algorithm.

# In[14]:


printdf(data)


# From the dataset above, we need to create multiple customer segments based on each user’s purchase behavior.
# 
# The variables in this dataset are in a format that cannot be easily ingested into the customer segmentation model. These features individually do not show us much about customer purchase behavior. Therefore, we would use the existing variables to derive three new informative features - recency, frequency, and monetary value (RFM).
# 
# RFM is commonly used in marketing to evaluate a client’s value based on their:
# 
# Recency: How recently has each customer made a purchase?
# Frequency: How often have they bought something?
# Monetary Value: How much money do they spend on average when making purchases?
# We will now preprocess the dataframe to create the above variables.

# In[15]:


# We will choose some suitable variables for RFM model:
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")

#Select the Quantity column with value > 0 to get the positive value
rtl_data = rtl_data.filter(F.col("Quantity") > 0)

#Calculate the TotalAmount by multiple Quantity with UnitPrice, serve for calculating Monetary value
rtl_data = rtl_data.withColumn("TotalAmount", F.round(F.col("Quantity") * F.col("UnitPrice")))

#Change the InvoiceDate column from datetime type to date type
rtl_data = rtl_data.withColumn("InvoiceDate",
                               F.to_date(F.col("InvoiceDate"), 'MM/dd/yyyy'))
printdf(rtl_data)


# In[16]:


#Check the earliest/latest time that customers go shopping:
df = rtl_data.toPandas()

print(df.InvoiceDate.min())
print(df.InvoiceDate.max())


# In[17]:


rtl_data.select(F.max('InvoiceDate')).collect()


# In[18]:


#Build the RFM model:
latest_date = F.to_date(F.lit("2011/12/10"), 'yyyy/MM/dd')
rfm_scores = (rtl_data.groupBy("CustomerID")
              .agg((F.datediff(latest_date, F.max(F.col("InvoiceDate")))).alias("Recency"),
                   F.count("*").alias("Frequency"),
                   F.sum(F.col("TotalAmount")).alias("Monetary")).sort("CustomerID"))


# In[19]:


rfm_scores = (
    rfm_scores.withColumn(
        "Monetary",
        F.when(F.col("Monetary") <= 0, 1)
         .otherwise(F.col("Monetary")))
)


# In[20]:


printdf(rfm_scores)


# ### Standardization
# 
# Before building the customer segmentation model, let’s standardize the dataframe to ensure that all the variables are around the same scale:

# In[21]:


from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

assemble=VectorAssembler(inputCols=[
    'Recency','Frequency','Monetary'
], outputCol='features')

assembled_data=assemble.transform(rfm_scores)

scale=StandardScaler(inputCol='features',outputCol='standardized')
data_scale=scale.fit(assembled_data)
data_scale_output=data_scale.transform(assembled_data)


# In[22]:


data_scale_output.select('standardized').show(2,truncate=False)


# ## 5. Building the K-Means Clustering model:

# Now that we have completed all the data analysis and preparation, let’s build the K-Means clustering model.
# 
# The algorithm would be created using PySpark’s machine learning API.

# In[23]:


from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import numpy as np

seed = 20

costs = {}

for k in range(2, 10):
    min_cost = float('inf')
    kmeans = KMeans(featuresCol='standardized', k=k, seed=seed, initSteps=5, maxIter=100, initMode='k-means||')
    model = kmeans.fit(data_scale_output)
    cost = model.summary.trainingCost
    if cost < min_cost:
        min_cost = cost
    costs[k] = min_cost

print(costs)


# In[24]:


k_values = list(costs.keys())
cost_values = list(costs.values())

plt.plot(k_values, cost_values, marker='o')
plt.xlabel('K')
plt.ylabel('Cost')
plt.title('Elbow Method')
plt.show()


# In[25]:


#Based on the plotted graph, we observe a distinct bend or "elbow" at the value of four.
#This suggests that there's a point of diminishing returns or significant change in the data's behavior around that cluster count.
#Consequently, we'll move forward with constructing the K-Means algorithm using four clusters.
KMeans_algo = KMeans(featuresCol='standardized', k=4, seed=seed)
KMeans_fit = KMeans_algo.fit(data_scale_output)


# In[26]:


preds=KMeans_fit.transform(data_scale_output)

preds.show(5,0)


# ## 6. Data Visualization and Insights:

# The goal of these scatter plots and column charts is to visualize the distribution of customers in terms of their Recency, Frequency, and Monetary values, and to see how these customers are grouped into different clusters based on the 'prediction' column.

# In[27]:


#To see the results more clearly, we will use column charts:
#The results from these charts will help us get and show insights below:
import matplotlib.pyplot as plt

preds_data = preds.select('CustomerID', 'Recency', 'Frequency', 'Monetary', 'prediction').toPandas()

colors = ['r', 'g', 'b', 'y']

attributes = ['Recency', 'Frequency', 'Monetary']

#The code you provided generates bar plots to visualize the average values of specific attributes for each cluster.
#Each bar plot represents the mean value of a particular attribute (e.g., Recency, Frequency, Monetary) for the four clusters.
for attribute in attributes:
    plt.figure(figsize=(8, 6))
    for cluster in range(4):
        cluster_data = preds_data[preds_data['prediction'] == cluster]
        cluster_attribute_mean = cluster_data[attribute].mean()
        plt.bar(cluster, cluster_attribute_mean, color=colors[cluster], label=f'Cluster {cluster}')
    plt.ylabel(attribute)
    plt.xticks(range(4), ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'])
    plt.legend()
    plt.show()


# Some insights about these specific clusters:
# - Cluster 0: Customers with medium recency, low frequency, and low monetary value are those who have made a purchase somewhat recently but do so infrequently and spend relatively little each time. This segment might represent occasional or casual buyers who have shown some recent interest in your offerings but aren't deeply engaged.
# - Cluster 1: In this cluster, customers who show high recency, do not visit the site often and have low monetary value. This indicates that although these customers have shown recent interest to buy products, their overall engagement is low and they do not contribute significantly to revenue.
# - Cluster 2: These customers show low recency, very high frequency and high monetary. This suggests that these customers were highly engaged and valuable but have not interacted with the site recently. They might be prime candidates for re-engagement campaigns to bring them back to the site.
# - Cluster 3: This segment comprises users who show medium recency and medium frequency but purchase on the platform with high monetary value. This indicates that they tend to buy high-value items or make bulk purchases. They may not be as engaged or loyal as other segments, but their high monetary value shows that this is an important customer segment to target and retain and improve our service to suit this cluster

# After doing the RFM Analysis, we can propose an allocation of cost percentages plan to each cluster. For this example, let's assume a total cost allocation of 100%.
# 
# - Cluster 0 (Medium Recency, Low Frequency, Low Monetary Value): Allocate 5-10% of cost to this cluster
# 
# 
# While these customers may not contribute significantly to revenue, a small portion of the budget is allocated to re-engage them and prevent further attrition
# 
# Activities for this cluster: Focus on improving customer engagement and increasing the average order value.
# Implement strategies to encourage repeat purchases, such as offering loyalty programs, personalized recommendations, or targeted promotions based on past purchase behavior.
# Enhance the user experience on the website to make it easier for customers to find products and complete purchases.
# Consider bundling products or offering discounts for bulk purchases to incentivize higher spending
# 
# - Cluster 1 (High Recency, Very Low Frequency, Very Low Monetary Value): Allocate 10-15% of cost to this cluster.
# 
# Since these customers spend relatively little and infrequently, a smaller portion of the budget is allocated to encourage them to increase their spending.
# 
# Activities for this cluster: Invest in strategies to increase customer engagement and retention.
# Develop personalized marketing campaigns to re-engage customers who have shown recent interest but haven't made frequent purchases.
# Offer incentives such as exclusive discounts, freebies, or referral programs to encourage these customers to visit the site more often and make purchases.
# Improve the overall customer experience to make it more compelling for them to return, such as optimizing website performance, providing excellent customer service, and streamlining the checkout process.
# 
# 
# - Cluster 2 (Low Recency, High Frequency, Medium Monetary Value): Allocate 30-35% of cost to this cluster
# .
# These engaged customers represent potential for increased spending, so a moderate portion of the budget is allocated to maintain their loyalty and encourage further purchases
# 
# Activities for this cluster: Launch targeted re-engagement campaigns to bring back these valuable customers who have previously shown high engagement and spending.
# Use data-driven insights to identify the reasons for their decreased interaction with the site and tailor offers or incentives to address their specific needs or preferences.
# Leverage email marketing, social media, and personalized messaging to reconnect with these customers and remind them of the value proposition of your products or services.
# Consider offering exclusive deals or perks to incentivize their return, such as limited-time promotions or VIP access to new products or events.
# 
# 
# - Cluster 3 (Medium Recency, Medium Frequency, High Monetary Value): Allocate 35-40% of cost to this cluster
# 
# 
# These high-value customers represent a significant portion of revenue, so a larger portion of the budget is allocated to maintain their loyalty and encourage continued high-value transactions
# 
# 
# These percentages are just rough estimates and can vary based on factors such as the size of each cluster, the potential for growth, and the company's overall revenue goal
# 
# Activities for this cluster: Prioritize customer satisfaction and retention strategies to capitalize on the high monetary value of this segment.
# Offer personalized services or incentives to enhance their loyalty and encourage repeat purchases, such as dedicated account managers, priority support, or exclusive access to premium content or events.
# Collect feedback regularly to understand their evolving needs and preferences and adapt your offerings accordingly.
# Explore opportunities for upselling or cross-selling to maximize the lifetime value of these customers, such as recommending complementary products or services based on their past purchases.
# 

# # III. Conclusion

# In conclusion, gaining a comprehensive understanding of the components within a firm, particularly its customers, is paramount for strategic success. In this project, businesses have been equipped with invaluable insights into their customer base through the implementation of the RFM model and the utilization of the K-means clustering algorithm with the Elbow method. The application of these models and algorithms in the project has not only provided businesses with a deeper understanding of their customers' characteristics, behaviors, and demographics but has also enabled them to identify distinct customer segments based on transaction data, thereby tailoring their strategies to meet the specific needs and preferences of each segment.
# 
# By leveraging these insights and the specific recommendations above, businesses can build targeted marketing campaigns, personalized product recommendations, and optimized pricing strategies, ultimately leading to enhanced customer satisfaction, increased sales, and improved competitiveness. Therefore, the integration of RFM techniques and clustering algorithms has proven to be instrumental in enabling businesses to develop highly effective and tailored strategies, driving sustainable growth and success in today's dynamic business landscape. We hope that the above experimental results provide businesses with further ideas and suggestions for developing their businesses. 
# 
