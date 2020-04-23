
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv("chennai_house_price_prediction.csv")


# In[9]:


data.head(10)


# In[5]:


#Relation b/w Price and Area


# In[8]:


plt.scatter(data["INT_SQFT"],data["SALES_PRICE"],c="r")


# In[10]:


data["BUILDTYPE"].value_counts()


# In[11]:


data["BUILDTYPE"].replace({"Other":"Others","Comercial":"Commercial"},inplace=True)


# In[12]:


fig,ax=plt.subplots()
color={"House":"blue","Commercial":"r","Others":"green"}
plt.scatter(data["INT_SQFT"],data["SALES_PRICE"],c=data["BUILDTYPE"].apply(lambda x: color[x]))


# In[13]:


#Relation b/w Sales price and no. of bedroom and bathroom


# In[17]:


data.pivot_table(values="SALES_PRICE",index="N_BEDROOM",columns="N_BATHROOM",aggfunc="median")


# In[26]:


fig,ax=plt.subplots(2,2)
fig.set_figheight(10)
fig.set_figwidth(10)
ax[0,0].scatter(data["QS_BATHROOM"],data["SALES_PRICE"])
ax[0,0].set_title("QS_BATHROOM")
ax[0,1].scatter(data["QS_BEDROOM"],data["SALES_PRICE"])
ax[0,1].set_title("QS_BEDROOM")
ax[1,0].scatter(data["QS_ROOMS"],data["SALES_PRICE"])
ax[1,0].set_title("QS_ROOMS")
ax[1,1].scatter(data["QS_OVERALL"],data["SALES_PRICE"])
ax[1,1].set_title("QS_OVERALL")


# In[30]:


fig=plt.figure()
ax=fig.add_subplot(111)
ax.set_title("Box plot for house quality")
bp=ax.boxplot([data["QS_BATHROOM"],data["QS_BEDROOM"],data["QS_ROOMS"],data["QS_OVERALL"]])


# In[32]:


data["QS_OVERALL"].isnull().sum()


# In[33]:


for i in range(0,len(data)):
    if pd.isnull(data["QS_OVERALL"][i])==True:
            data["QS_OVERALL"][i]=(data["QS_BATHROOM"][i]+data["QS_BEDROOM"][i]+data["QS_ROOMS"][i])/3


# In[34]:


fig=plt.figure()
ax=fig.add_subplot(111)
ax.set_title("Box plot for house quality")
bp=ax.boxplot([data["QS_BATHROOM"],data["QS_BEDROOM"],data["QS_ROOMS"],data["QS_OVERALL"]])


# In[35]:


data.groupby("BUILDTYPE").SALES_PRICE.median()


# In[37]:


temp=data.loc[(data["BUILDTYPE"]=="Commercial")&(data["AREA"]=="Anna Nagar")]
temp["SALES_PRICE"].plot.hist(bins=50)


# In[38]:


data["SALES_PRICE"].plot.hist(bins=100)


# In[39]:


temp=data.loc[(data["BUILDTYPE"]=="House")&(data["AREA"]=="Anna Nagar")]
temp["SALES_PRICE"].plot.hist(bins=50)


# In[40]:


#Relation with Parking facility


# In[46]:


data["PARK_FACIL"].replace({"Noo":"No"},inplace=True)
data["PARK_FACIL"].value_counts()


# In[47]:


data.groupby(["BUILDTYPE","PARK_FACIL"]).SALES_PRICE.median()


# In[53]:


temp=data.groupby(["BUILDTYPE","PARK_FACIL"]).SALES_PRICE.median()
temp.plot.bar()

