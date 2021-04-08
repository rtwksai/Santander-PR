#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


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
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


def rename_columns(data):
    data.rename(columns = {"fecha_dato":"time_series","ncodpers":"customer_code","ind_empleado":"employee_index",                       "pais_residencia":"country_residence","sexo":"gender","fecha_alta":"Date_First_Customer",                       "ind_nuevo":"New_Customer_ind","antiguedad":"Seniority","indrel":"primary_cust",                       "ult_fec_cli_1t":"last_date_primary","indrel_1mes":"customer_type","tiprel_1mes":"cust_rel_type",                       "indresi":"residence_index","indext":"foriegn_index","conyuemp":"spouse_index","canal_entrada":"channel_by_cust_joined",                       "indfall":"deceased_index","tipodom":"primary_address","cod_prov":"province_code","nomprov":"province_name",                       "ind_actividad_cliente":"activity_index","renta":"gross_income","segmento":"segmentation",                       "ind_ahor_fin_ult1":"savings_account","ind_aval_fin_ult1":"guarantees","ind_cco_fin_ult1":"current_account",                       "ind_cder_fin_ult1":"derivative_account","ind_cno_fin_ult1":"payroll_account","ind_ctju_fin_ult1":"jnr_account",                       "ind_ctma_fin_ult1":"mas_particular_account","ind_ctop_fin_ult1":"particular_account","ind_ctpp_fin_ult1":"particular_Plus_Account",                       "ind_deco_fin_ult1":"short_term_deposits","ind_deme_fin_ult1":"medium_term_deposits",                       "ind_dela_fin_ult1":"long_term_deposits","ind_ecue_fin_ult1":"e_account","ind_fond_fin_ult1":"funds",                       "ind_hip_fin_ult1":"mortgage","ind_plan_fin_ult1":"pensions","ind_pres_fin_ult1":"loans",                       "ind_reca_fin_ult1":"taxes","ind_tjcr_fin_ult1":"credit_card","ind_valo_fin_ult1":"securities",                       "ind_viv_fin_ult1":"home_account","ind_nomina_ult1":"payroll","ind_nom_pens_ult1":"pensions1",
                       "ind_recibo_ult1":"direct_debit"},inplace=True)


# In[ ]:


products = ["savings_account","guarantees","current_account","derivative_account",           "payroll_account","jnr_account","mas_particular_account","particular_account",           "particular_Plus_Account","short_term_deposits","medium_term_deposits","long_term_deposits",           "e_account","funds","mortgage","pensions",            "loans","taxes","credit_card","securities",            "home_account","payroll","pensions1","direct_debit"]


# In[ ]:


test = pd.read_csv("../input/santander-pr/test.csv")
compute_df = pd.DataFrame()
compute_df['ncodpers'] = test['ncodpers']


# In[ ]:


result = pd.DataFrame()
for p in products:
    #add path excluding the product name here
    df = pd.read_csv("../input/first-run-4lags-logistic/products_aggregate_1/"+p+"_xgresult.csv")
    result[p] = df[p]
result.head()


# In[ ]:


result.dtypes


# In[ ]:


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


compute_df.to_csv("finalDday.csv",index=False)

