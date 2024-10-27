#!/usr/bin/env python
# coding: utf-8

# # Final Project
# 
# ## Predict whether a mammogram mass is benign or malignant
# 
# We'll be using the "mammographic masses" public dataset from the UCI repository (source: https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass)
# 
# This data contains 961 instances of masses detected in mammograms, and contains the following attributes:
# 
# 
#    1. BI-RADS assessment: 1 to 5 (ordinal)  
#    2. Age: patient's age in years (integer)
#    3. Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
#    4. Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
#    5. Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
#    6. Severity: benign=0 or malignant=1 (binominal)
#    
# BI-RADS is an assesment of how confident the severity classification is; it is not a "predictive" attribute and so we will discard it. The age, shape, margin, and density attributes are the features that we will build our model with, and "severity" is the classification we will attempt to predict based on those attributes.
# 
# Although "shape" and "margin" are nominal data types, which sklearn typically doesn't deal with well, they are close enough to ordinal that we shouldn't just discard them. The "shape" for example is ordered increasingly from round to irregular.
# 
# A lot of unnecessary anguish and surgery arises from false positives arising from mammogram results. If we can build a better way to interpret them through supervised machine learning, it could improve a lot of lives.
# 
# ## Your assignment
# 
# Build a Multi-Layer Perceptron and train it to classify masses as benign or malignant based on its features.
# 
# The data needs to be cleaned; many rows contain missing data, and there may be erroneous data identifiable as outliers as well.
# 
# Remember to normalize your data first! And experiment with different topologies, optimizers, and hyperparameters.
# 
# I was able to achieve over 80% accuracy - can you beat that?
# 

# ## Let's begin: prepare your data
# 
# Start by importing the mammographic_masses.data.txt file into a Pandas dataframe (hint: use read_csv) and take a look at it.

# In[ ]:





# Make sure you use the optional parmaters in read_csv to convert missing data (indicated by a ?) into NaN, and to add the appropriate column names (BI_RADS, age, shape, margin, density, and severity):

# In[ ]:





# Evaluate whether the data needs cleaning; your model is only as good as the data it's given. Hint: use describe() on the dataframe.

# In[ ]:





# There are quite a few missing values in the data set. Before we just drop every row that's missing data, let's make sure we don't bias our data in doing so. Does there appear to be any sort of correlation to what sort of data has missing fields? If there were, we'd have to try and go back and fill that data in.

# In[ ]:





# If the missing data seems randomly distributed, go ahead and drop rows with missing data. Hint: use dropna().

# In[ ]:





# Next you'll need to convert the Pandas dataframes into numpy arrays that can be used by scikit_learn. Create an array that extracts only the feature data we want to work with (age, shape, margin, and density) and another array that contains the classes (severity). You'll also need an array of the feature name labels.

# In[ ]:





# Some of our models require the input data to be normalized, so go ahead and normalize the attribute data. Hint: use preprocessing.StandardScaler().

# In[ ]:





# ## Now build your neural network.
# 
# Now set up an actual MLP model using Keras:

# In[ ]:





# In[ ]:





# ## How did you do?
# 
# Which topology, and which choice of hyperparameters, performed the best? Feel free to share your results!

# In[ ]:




