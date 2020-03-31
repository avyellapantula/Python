#!/usr/bin/env python
# coding: utf-8

# In[49]:


get_ipython().system('jupyter nbconvert --to script A1_guide.ipynb')
# ignore this top script, it just converts things to .py files for me from notebook files
"""
Created on Fri Jul 26 18:35:01 2019
@author: avocado
Estimated Completion Time: 20 minutes
"""

"""
We're going to be working with dataframes
import the necessary modules:
numpy
pandas
"""
import numpy as np
import pandas as pd

"""
you'll only be using np and pd to refer, whenever u use these modules
if you don't say 'as np' or 'as pd', then when you refer to certain things, you'll
have to say numpy.arange(w.e goes here) instead of np.arange()
"""


# In[44]:


# we'll be using pandas to create dataframes and manipulate them and
# numpy to manipulate the data in them

# most np things deal with arrays and number stuff
zer = np.zeros(10)
onne = np.ones(10)
print(zer)
print(onne)
# arange is also useful for your arsenal
print(np.arange(3))
print(np.arange(3.))
print(np.arange(3,9))
# these are both 1d stuff. you can arrange
mult_arr = np.full((3, 4), 7)
print(mult_arr)


# In[47]:


# you can also create ur own lists
listy = []
# empty list
listy.append('Foop')
print(listy)
lister = ['a','b','c']
print(lister)
strin='string!!!'
lister.append(strin)
print(lister)

# numbers work too
numlist=[]
for i in range(12):
    numlist.append(i)
#     print(i)
print(numlist)
numlist.append(zer)


# In[37]:


# now lets mess around with pd
# normally i use csvs for pd, bc it's easy to manage,
# but you should know how to make one

# there are a couple ways to create a dataframe:
# list of lists
dat=[['A','12'],['B',45],['C',900]]
# if u print this as is, u'll get an array which looks ugly
# we want to convert this to a dataframe

# dat=pd.DataFrame(dat,columns=['Letter','Num'])

# now if u print it, it looks pretty (jupyter notebook)
# say u already had an array. how would u make that array the names of the columns?
d_col=['name1','name2']
dat=pd.DataFrame(dat,columns=d_col)
dat


# In[48]:


# the other way is as follows:
data = {'Yip':['A', 'Z', 'B', 'Y'], 'Nop':[0, 81, 9, 38]} 
  
# Create DataFrame 
df = pd.DataFrame(data) 
  
# Print the output. 
df 


# In[41]:


# suppose we have some other data to add
moredat=['foo',76,'noop']
# how to add column
dat['name3']=moredat
# dat
# now say we want to know what kind of table we have with us. what should we do?
# if we want to know datatypes:
# dat.dtypes or dat.info()

# note that while you only made a table with 3 columns, 
# you have a fourth at the beginning. That one's the index.
# If u want to use name1 as the index, here's what we have to do:

dat.set_index('name1',inplace=True)
dat

