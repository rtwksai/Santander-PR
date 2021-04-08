#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# In[ ]:


#Renaming columns for ease of use
def rename_columns(data):
    data.rename(columns = {"fecha_dato":"time_series","ncodpers":"customer_code","ind_empleado":"employee_index",                       "pais_residencia":"country_residence","sexo":"gender","fecha_alta":"Date_First_Customer",                       "ind_nuevo":"New_Customer_ind","antiguedad":"Seniority","indrel":"primary_cust",                       "ult_fec_cli_1t":"last_date_primary","indrel_1mes":"customer_type","tiprel_1mes":"cust_rel_type",                       "indresi":"residence_index","indext":"foriegn_index","conyuemp":"spouse_index","canal_entrada":"channel_by_cust_joined",                       "indfall":"deceased_index","tipodom":"primary_address","cod_prov":"province_code","nomprov":"province_name",                       "ind_actividad_cliente":"activity_index","renta":"gross_income","segmento":"segmentation",                       "ind_ahor_fin_ult1":"savings_account","ind_aval_fin_ult1":"guarantees","ind_cco_fin_ult1":"current_account",                       "ind_cder_fin_ult1":"derivative_account","ind_cno_fin_ult1":"payroll_account","ind_ctju_fin_ult1":"jnr_account",                       "ind_ctma_fin_ult1":"mas_particular_account","ind_ctop_fin_ult1":"particular_account","ind_ctpp_fin_ult1":"particular_Plus_Account",                       "ind_deco_fin_ult1":"short_term_deposits","ind_deme_fin_ult1":"medium_term_deposits",                       "ind_dela_fin_ult1":"long_term_deposits","ind_ecue_fin_ult1":"e_account","ind_fond_fin_ult1":"funds",                       "ind_hip_fin_ult1":"mortgage","ind_plan_fin_ult1":"pensions","ind_pres_fin_ult1":"loans",                       "ind_reca_fin_ult1":"taxes","ind_tjcr_fin_ult1":"credit_card","ind_valo_fin_ult1":"securities",                       "ind_viv_fin_ult1":"home_account","ind_nomina_ult1":"payroll","ind_nom_pens_ult1":"pensions1",
                       "ind_recibo_ult1":"direct_debit"},inplace=True)


# In[ ]:


main_train = pd.read_csv("../input/santander-pr/train.csv")
rename_columns(main_train)
main_train.dtypes
#Removing bad rows which have all attributes empty after verifying such row d.n.e. in test data
main_train = main_train[main_train['employee_index'].notna()]


# In[ ]:


products = ["savings_account","guarantees","current_account","derivative_account",           "payroll_account","jnr_account","mas_particular_account","particular_account",           "particular_Plus_Account","short_term_deposits","medium_term_deposits","long_term_deposits",           "e_account","funds","mortgage","pensions",            "loans","taxes","credit_card","securities",            "home_account","payroll","pensions1","direct_debit"]
non_pro = [x for x in main_train.columns if x not in products+['spouse_index','province_name','last_date_primary','province_name','customer_type','cust_rel_type','Date_First_Customer']]
non_pro = non_pro + ['Date_first_customer_year','time_series_month']
print(len(products),len(non_pro))


# In[ ]:


#Imputing based on what we presented in our initial presentations
def preprocessing_dat(data_given):
    data = data_given
    data['age'] = pd.to_numeric(data['age'])
    data['age_square'] = np.square(data['age'])
    data.loc[(data.age < 40) & (data.segmentation.isnull()),'segmentation'] = '03 - UNIVERSITARIO'
    data.loc[(data.age < 50) & (data.segmentation.isnull()) & (data.age >=40),'segmentation'] = '02 - PARTICULARES'
    data.loc[(data.segmentation.isnull()) & (data.age >=50),'segmentation'] = '01 - TOP'
    data.loc[data['province_code'].isnull(), 'province_code'] = 28.0
    data.gross_income = data.groupby('province_code')['gross_income'].apply(lambda x : x.fillna(x.median()))
    data['gross_income_log'] = np.log(data.gross_income)
    data.gross_income = data.gross_income.fillna(data.gross_income.median())
    data.loc[(data['gender'].isna()) & (data['customer_code']%2 == 0),'gender'] = 'H'
    data.loc[(data['gender'].isna()) & (data['customer_code']%2 == 1),'gender'] = 'V'
    data['Date_First_Customer'] = pd.to_datetime(data['Date_First_Customer'])
    data.channel_by_cust_joined = data.groupby(data.Date_First_Customer.dt.year)['channel_by_cust_joined'].apply(lambda x : x.fillna(x.mode()[0]))
    data.channel_by_cust_joined = data.channel_by_cust_joined.fillna(data.channel_by_cust_joined.mode()[0])
    data['Seniority'] = data['Seniority'].astype('int32')
    data['time_series'] = pd.to_datetime(data['time_series'])
    data['Date_First_Customer'] = pd.to_datetime(data['Date_First_Customer'])
    data['time_series_month'] = data['time_series'].apply(lambda x : x.month)
    data['time_series_month_sq'] = np.square(data['time_series'].apply(lambda x : x.month))
    data['Date_first_customer_year'] = data['Date_First_Customer'].apply(lambda x : x.year)
    data['Date_first_customer_month'] = data['Date_First_Customer'].apply(lambda x : x.month)
    data['cust_rel_type'] = data['cust_rel_type'].astype('str')
    data['cust_rel_type'].fillna(data['cust_rel_type'].mode()[0])
    data.drop(columns=['spouse_index','province_name','last_date_primary','customer_type','Date_First_Customer'],inplace=True)
    return data


# In[ ]:


#Few product columns are null we will them with 0's because their majority in every product
def preprocess_products(data_given):
    data = data_given
    data.payroll = data.payroll.fillna(0)
    data.pensions = data.pensions.fillna(0)
    data.pensions1 = data.pensions1.fillna(0)
    return data


# In[ ]:


#Customised lag function where we can specify how months lag and what attributes lag do we need
def lagn(df,n=4,cols=products):
    time_groups = df.groupby('time_series')
    dfs = []
    for time in time_groups.groups:
        mini_df = time_groups.get_group(time)
        for off in range(1,n+1):
            prev_month = time - pd.DateOffset(months=off)
            lag_products = main_train.loc[main_train['time_series']==prev_month,cols+['customer_code']]
            if(off == 1):
                lag_names = [x + '_lag' for x in cols]
            else:
                lag_names = [x + '_lag'+str(off) for x in cols]
            rename_col = {cols[i]: lag_names[i] for i in range(len(cols))}
            lag_products.rename(columns=rename_col,inplace=True)
            mini_df = pd.merge(mini_df,lag_products,on='customer_code',how='left')
        mini_df.fillna(0,inplace=True)
        dfs.append(mini_df)
    resultant_df = pd.concat(dfs)
    resultant_df.drop(columns=['time_series','customer_code'],inplace=True)
    return resultant_df


# In[ ]:


#Customised exponential smoothing addition for the passed df
alpha_columns = ['alpha_0.03','alpha_0.1','alpha_0.3','alpha_0.9']
alpha_vals = np.array([0.03,0.1,0.3,0.9])

def alpha_lag(df,cols=alpha_columns,cur_product='savings_account'):
    time_groups = df.groupby('time_series')
    dfs = []
    for time in time_groups.groups:
        mini_df = time_groups.get_group(time)
        prev_month = time - pd.DateOffset(months=1)
        lag_products = main_train.loc[main_train['time_series']==prev_month,cols+['customer_code']]
        lag_names = [x + '_lag' for x in cols]
        rename_col = {cols[i]: lag_names[i] for i in range(len(cols))}
        lag_products.rename(columns=rename_col,inplace=True)
        mini_df = pd.merge(mini_df,lag_products,on='customer_code',how='left')
        mini_df.fillna(0,inplace=True)
        for i in range(len(cols)):
            mini_df[cols[i]] = alpha_vals[i]*mini_df[cur_product+'_lag'] + (1-alpha_vals[i])*mini_df[cols[i]+'_lag']
        dfs.append(mini_df)
    resultant_df = pd.concat(dfs)
    resultant_df.drop(columns=['time_series','customer_code'],inplace=True)
    return resultant_df


# In[ ]:


main_train = preprocessing_dat(main_train)
main_train = preprocess_products(main_train)


# In[ ]:


#Place where we tried sampling various months from the main train
train_sample = main_train.loc[(main_train.time_series == '2016-03-28')]


# In[ ]:


#Adding 6 lags of six month data to the provided df
train_data = lagn(train_sample,n=6)


# In[ ]:


#Saving memory by removing unused variables
del train_sample
main_train = main_train.loc[main_train.time_series > pd.to_datetime('2015-10-28')]


# In[ ]:


test = pd.read_csv("../input/santander-pr/test.csv")
rename_columns(test)
#Final df that we will be writing to csv, storing the id's in the order of test to avoid confusion
compute_df = pd.DataFrame()
compute_df['ncodpers'] = test['customer_code']


# In[ ]:


test = preprocessing_dat(test)
#introducing same lag in test data
test = lagn(test,n=6)

train_dat = 0
#Clearing main_train for saving memory
del main_train #Thank you main train


# In[ ]:


#Block for encoding test and train samples

obj_col = list(train_data.select_dtypes(['object']).columns) #obj_columns
le = preprocessing.LabelEncoder()
train_data[obj_col] = train_data[obj_col].apply(le.fit_transform)
train_data.head()
train_data.info()
test[obj_col] = test[obj_col].apply(le.fit_transform)


# In[ ]:


train_data.drop(columns=['time_series','customer_code'],inplace=True)
test.drop(columns=['time_series','customer_code'],inplace=True)


# In[ ]:


#Block to review if we can remove any unused variables in RAM
from __future__ import print_function 
import sys

local_vars = list(locals().items())
for var, obj in local_vars:
    print(var, sys.getsizeof(obj))


# In[ ]:


#Min Max scalar useful only if we use tradional ml techniques though
x = train_data.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
train_data = pd.DataFrame(x_scaled,columns=train_data.columns,index=train_data.index)
#Scaling for test data
x = test.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
test = pd.DataFrame(x_scaled,columns=test.columns,index=test.index)


# In[ ]:


#Dividing train and test attributes
y = train_data[products]
mask = train_data.columns.isin(products)
non_prod_cols = [item for item in train_data.columns if item not in products]
X = train_data[non_prod_cols]


# In[ ]:


#Removing unsued variables to make better use of memory
del train_data
del x_scaled
X.head()


# In[ ]:


'''Unused set of models
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler'''
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#Dictionary for storing probabilities of class predictions of validation set
probabilities = dict()
#Dictionary for storing probabilities of predictions of validation set
predictions = dict()
#Dictionary for storing models trained specific to each product
models = dict()

tscv = TimeSeriesSplit(n_splits=5)
for tr_index, val_index in tscv.split(X):
    X_tr, X_val = X.iloc[tr_index], y.iloc[tr_index]
    y_tr, y_val = X.iloc[val_index], y.iloc[val_index]
    
for category in products:
    print('**Processing {} product ...**'.format(category))
    clf = XGBClassifier(sampling_method='gradient_based',eta = 0.2, max_depth = 10, verbosity=2, gamma=10)
    clf.fit(X_tr, X_val[category])
    prediction = clf.predict(y_tr)
    predict_probability = clf.predict_proba(y_tr)
    predictions[category] = prediction
    probabilities[category] = predict_probability
    models[category] = clf
    print('Probability {} accuracy is {}'.format(category,predict_probability))
    print("\n")
    print('Test accuracy is {}'.format(accuracy_score(y_val[category], prediction)))
    print("\n")    


# In[ ]:


probabilities


# In[ ]:


test_prediction = {}
test_probas = {}
for category in products:
    prediction = models[category].predict(test)
    predict_probability = models[category].predict_proba(test)
    test_prediction[category] = prediction
    test_probas[category] = predict_probability[:,1]
    print('Probability {} is {}'.format(category,predict_probability))
    print("\n")


# In[ ]:


#Making a df from the probabilities
results_df = pd.DataFrame(test_probas)
results_df.head()


# In[ ]:


result = pd.DataFrame(abs(results_df.values - test[products_lag].values))
result.head()


# In[ ]:


#This function will be called inside pd.apply() to parallely compute top 5 suggests
def top5(row):
    product_name = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1',               'ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1',               'ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',               'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1',               'ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1',               'ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
    width = len(product_name)
    sort_index = np.argsort(row)
    sort_index = sort_index[::-1]
    product_list = [product_name[k] for k in sort_index[:5]]
    recom_string = ' '.join(product_list)
    return recom_string


# In[ ]:


compute_df['changed'] = result.apply(top5,axis=1)
compute_df.head()


# In[ ]:


compute_df.to_csv('final_lag.csv',index=False)

