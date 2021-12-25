# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import pycountry_convert as pc
from scipy.stats import pearsonr
import streamlit as st
import base64

# Creating a function that:
# 1. Aggregate State-wise data into Country Level
# 2. Dis-aggregate the Country Level data from accumulated totals to daily numbers
# 3. Pivot the Tables into a form that is suitable for analysis on Python

@st.cache
def dataframes_analysis(data):

    Country_names = []
    Daily_cases_d = []
    Daily_cases_c = []
    Error = []
    Continent_name = []
    Country_name = []

    for i in np.unique(data['Country/Region']):
        Country_names.append(i)
        cum_sum = np.sum(data[data['Country/Region'] == i].iloc[:,4:])
        Daily_cases_c.append(cum_sum)
        Daily_cases_d.append(np.r_[[0],np.diff(cum_sum)])

        # Error Handling 

    if len(Country_names) == len(Daily_cases_c) == len(Daily_cases_d):
        print ('')
        print ('Successfully aggregated records to Country Level and Disaggregated Cumulative daily records to Daily records')

    elif len(Country_names) != len(Daily_cases_c):
        print ('')
        print ('Error: Check step where records were aggregated to country level')

    elif len(Daily_cases_c) != len(Daily_cases_d):
        print ('')
        print ('Error: Check step where records were aggregated to country level')

    elif (Country_names) != len(Daily_cases_c) and len(Daily_cases_c) != len(Daily_cases_d):
        print ('')
        print ('Error in both aggregation of records to Country Level and disaggregation of records from Cumulative daily records to Daily records: Check all steps')

    # Preparing data in the Pivot format
    Country_names_repeated = []
    
    for i in Country_names:
        
        for j in range (len(Daily_cases_d[0])):

            Country_names_repeated.append(i)

    # Error Handling 
    if len(Country_names_repeated) / (len(Country_names) * len(Daily_cases_d[0])) != 1.0:
        print ('')
        print ('Error: The length of the extended Country_names list is not equal to len(Country_names) * Number of days')
    else:
        print ('')
        print ('Data ready for DataFrame creation')

    Case_table = pd.DataFrame({'Date':list(data.columns[4:])*len(Country_names),'Country':Country_names_repeated,'Daily':np.array(Daily_cases_d).ravel(), 'Agg_Daily':np.array(Daily_cases_c).ravel()})

    Case_table['Date'] = pd.to_datetime(Case_table['Date'])
    
    Case_table = Case_table[(Case_table.Country !='Summer Olympics 2020') & (Case_table.Country !='Diamond Princess') & (Case_table.Country != 'MS Zaandam')]
    
    
    # Creating column with the continents

    for i in np.unique(Case_table.Country): 

        try:

            country_code = pc.country_name_to_country_alpha2(i, cn_name_format="default")

            continent_name = pc.country_alpha2_to_continent_code(country_code)

            Continent_name.append(continent_name)

            Country_name.append(i)
            
        except:

            Error.append(i)
        
        
    Country_continent_dict = dict({(Country_name[i],Continent_name[i]) for i,j in enumerate(Country_name)})
    
    # The automatic continent finder function doesn't work on the following countries. Hence it will be updated manually
    
    Country_continent_dict_error = {'Korea, South': 'AS','Taiwan*': 'AS','Timor-Leste': 'AS','West Bank and Gaza': 'AF','US': 'NA','Congo (Kinshasa)': 'AF',
     'Congo (Brazzaville)': 'AF','Burma': 'AS','Kosovo': 'EU',"Cote d'Ivoire": 'AF','Holy See': 'EU'}
    
    #Updating the meta dictionary that contains the Continent details for each country
    Country_continent_dict.update(Country_continent_dict_error)
    
    
    # Creating a column with the Continent
    Case_table['Continent'] = Case_table.Country.map(Country_continent_dict)

    return Case_table


# Creating a function that enables the computation of correlations

def corr(cases,deaths,level,input_data,number_of_days):
    
    if level == 'World':
        
        A = deaths.groupby('Date').sum()['Daily'].rolling(number_of_days).corr(cases.groupby('Date').sum()['Daily'])
    
    elif level == 'Continent':
        
        A = deaths[deaths.Continent == input_data].groupby('Date').sum()['Daily'].rolling(number_of_days).\
        corr(cases[cases.Continent == input_data].groupby('Date').sum()['Daily'])
        
    elif level == 'Country':
        
        A = deaths['Daily'][deaths.Country == input_data].rolling(number_of_days).\
        corr(cases['Daily'][cases.Country == input_data])
        
    A = A.tolist()
    
    A_df = pd.DataFrame({'Date':np.unique(cases.Date), 'Correlation': A})
        
    return A_df


# Creating a new function for CAGR
# Function that computes the cagr
def cagr(input_type,level,input_data,number_of_days):
    
    if level == 'World':
        
        A = np.power((input_type.groupby('Date')['Daily'].sum().pct_change(number_of_days) + 1), 1/number_of_days)-1
    
    elif level == 'Continent':
        
        A = np.power((input_type[input_type.Continent == input_data].groupby('Date')['Daily'].sum().pct_change(number_of_days) + 1),\
                     1/number_of_days)-1
        
    elif level == 'Country':
        
        A = np.power((input_type[input_type.Country == input_data].groupby('Date')['Daily'].sum().pct_change(number_of_days) + 1),\
                     1/number_of_days)-1
        
    A = A.tolist()
    
    A_df = pd.DataFrame({'Date':np.unique(input_type.Date), 'CAGR': A})
        
    return A_df
    

