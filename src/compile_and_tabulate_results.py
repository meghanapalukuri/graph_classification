# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 01:55:48 2021

@author: Meghana
"""

from glob import glob
import pandas as pd

df_list = []
allsubd = '../results/*/results.csv'

for fname in glob(allsubd, recursive=True):
    df_list.append(pd.read_csv(fname,index_col=0))
    
allsubd = '../results/*/*/results.csv'

for fname in glob(allsubd, recursive=True):
    df_list.append(pd.read_csv(fname,index_col=0))
    
summarized_file = '../results/compiled_results.csv'

df_compiled = pd.concat(df_list)
df_compiled.sort_values(by='Validation accuracy',ascending=False,inplace=True)

df_compiled.to_csv(summarized_file)
    
    