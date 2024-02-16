#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Amazon sales data.csv')
df.head()


# ### Data cleaning

# In[3]:


sns.heatmap(df.isnull())
plt.show()


# In[5]:


df.dropna(inplace=True)
print(df.duplicated())


# In[16]:


df.columns


# In[18]:


df.shape


# In[2]:


df['discounted_price'] = df['discounted_price'].apply(lambda x: x.replace('₹',''))
df['actual_price'] = df['actual_price'].apply(lambda x: x.replace('₹',''))
df['discount_percentage'] = df['discount_percentage'].apply(lambda x: x.replace('%',''))
#df['actual_price'] = df['actual_price'].apply(lambda x: x.replace(',',''))
df.head()


# In[3]:


def replace_comma(df,columns):
    for col in columns:
        df[col] = df[col].str.replace(',','')
    
columns = ['actual_price', 'rating','rating_count', 'discounted_price']
replace_comma(df,columns)
#df.info()


# In[4]:


def change_type(df,columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = np.round(df[col].fillna(0)).astype('Int64')
columns  = ['actual_price','rating_count', 'discounted_price','discount_percentage']
change_type(df, columns)


# In[5]:


df['discount_percentage'] = df['discount_percentage'] / 100

df['rating'] = df['rating'].str.replace('|','0')
df['rating'] = df['rating'].astype(float)

df.info()


# ### Data analysis and interpretation

# In[34]:


sns.scatterplot(data = df,x='actual_price',y = 'discounted_price',color='c',marker='*').set_title('Actual Price vs Discounted Price')
plt.xlabel("Actual Price")
plt.ylabel("Discounted Price")
plt.show()


# In[14]:


sns.catplot(data=df,x='actual_price', hue='rating',marker='D')
plt.title("Rating based on Actual price")
plt.xlabel("Actual price")
plt.ylabel("Rating")
plt.show()


# In[12]:


sns.boxplot(data=df,x='actual_price', hue='rating_count')
plt.title("Rating count based on Actual price")
plt.xlabel("Actual price")
plt.ylabel("Rating-count")
plt.show()


# In[15]:


sns.stripplot(data=df,x='discounted_price', hue='rating')
plt.title("Rating based on discounted_price")
plt.xlabel("discounted_price")
plt.ylabel("Rating")
plt.show()


# In[17]:


sns.boxenplot(data=df,x='discounted_price', hue='rating_count')
plt.title("Rating count based on Actual price")
plt.xlabel("discounted_price")
plt.ylabel("Rating-count")
plt.show()


# In[7]:


sns.boxenplot(data=df,y='discount_percentage',hue='category')
plt.title("discount percentage according to  category")
plt.xlabel("category")
plt.ylabel("discount_percentage")
plt.show()


# In[14]:


category_counts = df['category'].value_counts().sort_values(ascending = False)
top10_categories = category_counts.head(10)

plt.barh(top10_categories.index, top_10_categories.values)
plt.xlabel('Count')
plt.ylabel('Category')
plt.title('Top 10 Categories by Frequency Count')
plt.show()


# In[25]:


df.tail(10)


# In[6]:


category_counts = df['category'].value_counts().sort_values(ascending = False)
least_categories = category_counts.tail(10)

plt.barh(least_categories.index, least_categories.values)
plt.xlabel('Count')
plt.ylabel('Category')
plt.title('Least 10 Categories by Frequency Count')
plt.show()


# In[42]:


rating_counts = df['rating'].value_counts().sort_values(ascending = False)
top_10_categories = category_counts.head(10)

plt.barh(top_10_categories.index, top_10_categories.values)
plt.xlabel('Count')
plt.ylabel('rating')
plt.title('Top 10 rating by Frequency Count')
plt.show()

