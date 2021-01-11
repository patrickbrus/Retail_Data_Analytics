# Retail_Data_Analytics
A Machine Learning project for retail data analytics as part of the Machine Learning Engineering Nanodegree Capstone Project from Udacity. 
The project is inspired by the Kaggle challenge on https://www.kaggle.com/manjeetsingh/retaildataset where also the used data can be downloaded.
My article about this project can be found on medium (https://towardsdatascience.com/retail-data-analytics-1391284ec7b8).

# Domain Background 
Retail Data Analytics (RDA) is used nowadays from shops in order to better predict the amount of articles, that might get sold and therefore to better estimate how much articles should be produced. This is very important, because the amount of sold articles can vary largely during the year. For example people are tend to buy more things before Christmas then during a normal, not holiday, week. This can be easily seen on the Amazon quarterly revenue on Statista (https://www.statista.com/statistics/273963/quarterly-revenue-of-amazoncom/). The quarterly revenue of Amazon is always the largest for the fourth quarter, which indicates, that the people are consuming more during the fourth quarter than during the others. This is clear due to the fact that Christmas is within the fourth quarter and also the Black Friday, which leads to large worldwide consume, too. If a shop has too few products before Christmas, he would loose potential income. But if a shop has too much products, too much storage would be required and storage also costs money, so the company would again loose money. RDA can therefore be used in order to try to optimize the production of products,such that there is always an optimal amount available.

# Problem Statement
The goal is to predict the department wide sales for each store for the following year. This should then help to optimize the manufacturing process and therefore to increase income while lowering costs. It should be possible to feed in past sales data from a department and to get the predicted sales for the following year.

# Description of Files
All notebooks were developed within the AWS Sagemaker environment.
1. The notebook 1_Data_Exploration.ipynb contains some code for the data analysis of the dataset.
2. The notebook 2_Create_Train_and_Test_Data.ipynb contains the code for merging all data together and creating the final csv files for training and testing.
3. The folder Documentation contains the Proposal for this Project.

# Libraries Used
This chapter briefly describes the main libraries that were used for this project. 
The main libraries are:
1. Scikit-Learn Library (https://scikit-learn.org/stable/).
2. Matplotlib Library (https://matplotlib.org).
3. Scikit-Optimize Library (https://scikit-optimize.github.io/stable/).
4. Seaborn (https://seaborn.pydata.org).
5. Sagemaker Python SDK (https://sagemaker.readthedocs.io/en/stable/).
