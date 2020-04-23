
# coding: utf-8

# In[2]:


import pandas as pd 
import matplotlib.pyplot as plt


# In[3]:


data=pd.read_csv("chennai_house_price_prediction.csv")


# In[4]:


data.head()


# ### Drop Duplicates (if any)

# In[5]:


data.drop_duplicates()


# In[6]:


data.drop_duplicates(subset="AREA")


# In[7]:


data.drop_duplicates(subset="AREA").shape


# In[8]:


data.shape


# ### Handling missing values

# In[9]:


#Drop rows having missing values


# In[10]:


data.dropna(axis=0,how="any")


# In[11]:


#Drop columns having missing values
data.dropna(axis=1,how="any")


# In[12]:


#but this results in data loss


# In[13]:


#Imputing missing values with mean , median or mode


# In[14]:


data["N_BEDROOM"].isnull().sum()


# In[15]:


data["N_BEDROOM"].loc[data["N_BEDROOM"].isnull()==True]


# In[16]:


data["N_BEDROOM"].fillna(value=data["N_BEDROOM"].mode()[0],inplace=True)


# In[17]:


data["N_BEDROOM"].isnull().sum()


# In[18]:


data["N_BATHROOM"].loc[data["N_BATHROOM"].isnull()==True]


# In[19]:


for i in [70,5087,6134,6371,6535]:
    if data.loc[i,"N_BEDROOM"]==1:
        data["N_BATHROOM"][i]=1
    else:
        data["N_BATHROOM"][i]=2


# In[20]:


data.loc[data["N_BATHROOM"].isnull()==True]


# In[21]:


data.loc[data["QS_OVERALL"].isnull()==True]


# In[22]:


data[["QS_ROOMS","QS_BATHROOM","QS_BEDROOM","QS_OVERALL"]].head()


# In[23]:


avg=(data["QS_BATHROOM"]+data["QS_BEDROOM"]+data["QS_ROOMS"])/3


# In[24]:


data.insert(column="Avg",value=avg,loc=17)


# In[25]:


data.head()


# In[26]:


del data["Avg"]


# In[27]:


import numpy as np


# In[28]:


for i in range(0,len(data)):
    if pd.isnull(data["QS_OVERALL"][i])==True:
            data["QS_OVERALL"][i]=(data["QS_BATHROOM"][i]+data["QS_BEDROOM"][i]+data["QS_ROOMS"][i])/3


# In[29]:


data.loc[data["QS_OVERALL"].isnull()==True]


# In[30]:


data.dtypes


# In[31]:


data=data.astype({"N_BEDROOM":"int64","N_BATHROOM":"int64"})


# In[32]:


data.dtypes


# ### Handling speeling errors

# In[33]:


l=["AREA","SALE_COND","PARK_FACIL","BUILDTYPE","UTILITY_AVAIL","STREET","MZZONE"]


# In[34]:


for i in l:
    print("***********Value count for "+i+"*************")
    print(data[i].value_counts())
    print()


# In[35]:


data["STREET"].replace({"NoAccess":"No Access"},inplace=True)
data["STREET"].value_counts()


# In[36]:


data["UTILITY_AVAIL"].replace({"All Pub":"AllPub"},inplace=True)
data["UTILITY_AVAIL"].value_counts()


# In[37]:


data["BUILDTYPE"].replace({"Comercial":"Commercial"},inplace=True)
data["BUILDTYPE"].value_counts()


# In[38]:


data["PARK_FACIL"].replace({"Noo":"No"},inplace=True)
data["PARK_FACIL"].value_counts()


# In[39]:


data["SALE_COND"].replace({"Adj Land":"AdjLand","Ab Normal":"AbNormal","Partiall":"Partial","PartiaLl":"Partial"},inplace=True)
data["SALE_COND"].value_counts()


# In[40]:


data["AREA"].replace({"Chrompt":"Chrompet","Chormpet":"Chrompet","Chrmpet":"Chrompet","TNagar":"T Nagar","Ana Nagar":"Anna Nagar","Karapakam":"Karapakkam","Ann Nagar":"Anna Nagar","Velchery":"Velachery","Adyr":"Adyar","KKNagar":"KK Nagar"},inplace=True)
data["AREA"].value_counts()


# In[41]:


data.isnull().sum()


# In[42]:


data.dtypes


# In[43]:


plt.scatter(data["SALE_COND"],data["SALES_PRICE"])


# In[44]:


data["BUILDTYPE"].value_counts()


# In[45]:


data["BUILDTYPE"].replace({"Other":"Others"},inplace=True)


# In[46]:


data["BUILDTYPE"].replace({"Others":0,"House":1,"Commercial":2},inplace=True)


# In[47]:


data=data.astype({"BUILDTYPE":"int64"})


# In[48]:


x=data["INT_SQFT"].values.reshape(-1,1)


# In[49]:


y=data["SALES_PRICE"]


# In[50]:


from sklearn.model_selection import train_test_split


# In[51]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.3,random_state=1)


# In[52]:


from sklearn import linear_model


# In[53]:


reg=linear_model.LinearRegression()


# In[54]:


reg.fit(xtrain,ytrain)


# In[55]:


reg.coef_


# In[56]:


reg.score(xtest,ytest)


# In[57]:


plt.scatter(reg.predict(xtrain),reg.predict(xtrain)-ytrain,color='red',label="Train Data")
plt.scatter(reg.predict(xtest),reg.predict(xtest)-ytest,color='blue',label="Test Data")
plt.legend()


# In[58]:


reg.predict([[2000]])


# In[59]:


ytest


# In[60]:


from sklearn.metrics import mean_absolute_error as mae


# In[61]:


error=mae(reg.predict(xtest),ytest)


# In[62]:


print(error)


# In[63]:


plt.plot(xtest,reg.predict(xtest),c="Black")


# In[64]:


x=data[["INT_SQFT","BUILDTYPE"]]
y=data["SALES_PRICE"]


# In[65]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.3,random_state=1)


# In[66]:


reg.fit(xtrain,ytrain)


# In[67]:


reg.coef_


# In[68]:


reg.score(xtest,ytest)


# In[69]:


plt.style.use("fivethirtyeight")
plt.scatter(reg.predict(xtrain),reg.predict(xtrain)-ytrain,color='red',label="Train Data")
plt.scatter(reg.predict(xtest),reg.predict(xtest)-ytest,color='blue',label="Test Data")
plt.legend()


# In[70]:


reg.predict([[2000,2]])


# In[71]:


error=mae(reg.predict(xtest),ytest)
print(error)

