# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from scipy.stats import pearsonr
import streamlit as st
import base64
import data_prep
import Plots_new
import sqlite3

# Page Layout
st.set_page_config(layout="wide")

st.sidebar.markdown('<h2 align="center", style="color:#86BC24"> <u> COVID-19 Analytics Dashboard </u> </h2>',unsafe_allow_html=True)

# This function converts images into a form that can be stored in an html tag
def img_to_bytes(img_path):
    with open(img_path, 'rb') as i:
        img_bytes = i.read()
    encoded_img = base64.b64encode(img_bytes).decode()
    return encoded_img

# COVID image
image_base64_vid = img_to_bytes('.//Siemens_COVID-19-Hero-Still-Solo-no-Background.png')

html_vid = f" <div style='width:100%;text-align:center;'> <img src='data:image/png;base64,{image_base64_vid}' width = '305' height = '180' ></div>"

st.sidebar.markdown(html_vid, unsafe_allow_html = True)

# Linkedin
image_base64_l = img_to_bytes('.//download.png')

link_linkedin = 'https://www.linkedin.com/in/ibrahim-hameem-65b57096'

html_linkedin = f" <div style='width:100%;text-align:center;'> <a href='{link_linkedin}' target= 'blank'> <img src='data:image/png;base64,{image_base64_l}' width = '35' height = '35' ></a> </div>"


# Twitter
image_base64_t = img_to_bytes('.//download (1).png')

link_twitter = 'https://twitter.com/HameemIbrahim'

html_twitter = f" <div style='width:100%;text-align:center;'> <a href='{link_twitter}' target= '_blank'> <img src='data:image/png;base64,{image_base64_t}' width = '35' height = '40' ></a> </div>"

st.sidebar.markdown(' <h3 align="center", style="color:#86BC24">Creator: M. Ibrahim Hameem</h3> ',unsafe_allow_html=True)

# Handles

col1,col2 = st.sidebar.columns(2)

with col1:
    
    st.markdown(html_linkedin, unsafe_allow_html=True)

with col2:
    st.markdown(html_twitter, unsafe_allow_html=True)

# About the creator
image_base64_ibz = img_to_bytes('.//ibz.png')

html_ibz = f" <div style='width:100%;text-align:center;'> <img src='data:image/png;base64,{image_base64_ibz}' width = '250' height = '250' ></div>"


with st.expander("About the creator"):

    col11, col22, col33, col44 = st.columns(4)

    with col11:

        st.markdown(html_ibz, unsafe_allow_html=True)

    with col22:

        st.markdown('<p> Data Science Enthusiast || MSc. Data Science and Artificial Intelligence (In progress) || Ex-Investment Banking - Associate || University of London Graduate - BSc. Mathematics and Economics (First Class) </p>', unsafe_allow_html=True)

        st.markdown('<p align = "justify"> I am a Data Science Enthusiast, currently pursuing an MSc. Data Science and Artificial Intelligence from the University of London. I am an Ex-Investment Banking Associate, who decided to take a few months off full-time work in order to focus on my master`s degree, whilst exploring new employment opportunities that are aligned with my current interest in Data Science and AI, and that builds on my domain experience in Investment Banking. <br> <br> As an Associate who was primarily engaged in Mergers and Acquisitions, I have been a part of several M&A and Advisory mandates across multiple industries including Financial Services, FMCG, Health Care, and Construction Materials. In addition, I also worked on generating prospective business development leads <br> <br>Over the past few months, I have been exploring and experimenting with a variety of Machine Learning and Deep Learning algorithms, applications, and projects. In addition, I have made it a point to accumulate a sound theoretical understanding of these algorithms. It is definitely a field that I thoroughly enjoy, and I am looking forward to pursuing a career within it! <br> </p>', unsafe_allow_html=True)


# Importing the datasets


# Global Cases:

# Connect the database
conn = sqlite3.connect('.//covid_dash_27_Dec_21.db3')

#Read the data in the form of a dataframe
Global_cases = pd.read_sql("SELECT * FROM Global_cases;", con=conn)

# Updating the Date column into datetime
Global_cases.Date = pd.to_datetime(Global_cases.Date)

#Close the connection to the database
conn.close()

#Global Deaths:

# Connect the database
conn = sqlite3.connect('.//covid_dash_27_Dec_21.db3')

#Read the data in the form of a dataframe
Global_deaths = pd.read_sql("SELECT * FROM Global_deaths;", con=conn)

# Updating the Date column into datetime
Global_deaths.Date = pd.to_datetime(Global_deaths.Date)

#Close the connection to the database
conn.close()


# Dictionary of Continents
continents = {'NA': 'North America','SA': 'South America', 'AS': 'Asia','OC': 'Oceania',
'AF': 'Africa','EU': 'Europe', 'World': 'World'}

continents_reverse = dict({(j,i) for i,j in continents.items()})

# Plotting 

True_False_dict = {'Yes':True, 'No':False}

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

# Sidebar toggles

# Default values

result = 'No'

result_11 = 'World'

result_12 = 'Canda'

number_of_days = 7

# Toggles

st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')

show_toggles = st.sidebar.select_slider('Show Toggles', ['No', 'Yes'])

if True_False_dict[show_toggles] == True:

    st.sidebar.markdown(" <u> **Cumulative + Daily: Cases and Deaths** </u>",unsafe_allow_html = True)

    result_11 = st.sidebar.selectbox('Select Continent / World',list(continents.values()))

    result = st.sidebar.radio('Show Country Analysis', options=['Yes','No'])

    if result == 'No':

        result_12 = None

    if True_False_dict[result] == True:

        if result_11 == 'World':

            result_12 = st.sidebar.selectbox('Select Country', np.unique(Global_cases.Country))

        else:
            result_12 = st.sidebar.selectbox('Select Country', np.unique(Global_cases[Global_cases.Continent == continents_reverse[result_11]].Country))

    else:

        pass

    st.sidebar.markdown(" <u> **Further Analysis** </u>",unsafe_allow_html = True)   

    number_of_days = st.sidebar.number_input("No.of days: Rolling Beta (Slope) and CAGR's",min_value=7)

#Plots

# Cumulative cases

st.markdown('<h3 align="center", style="color:darkolivegreen"> <u>  Cumulative Cases and Cumulative Deaths </u> </h3> ', unsafe_allow_html = True)

fig_1 = Plots_new.Cases_deaths(func1=Plots_new.plot1, func2=Plots_new.plot2, cases = Global_cases, 
deaths=Global_deaths, Country = result_12, Select= continents_reverse[result_11], type_of_graph='Agg_Daily', show_country=True_False_dict[result])

st.plotly_chart(fig_1)

st.markdown('')

# Daily Cases

st.markdown('<h3 align="center", style="color:darkolivegreen"> <u>  Daily Cases and Daily Deaths </u> </h3> ', unsafe_allow_html = True)

fig_2 = Plots_new.Cases_deaths(func1=Plots_new.plot1, func2=Plots_new.plot2, cases = Global_cases, 
deaths=Global_deaths, Country = result_12, Select= continents_reverse[result_11], type_of_graph='Daily', show_country=True_False_dict[result])

st.plotly_chart(fig_2)

st.markdown('')

# Further Analysis

number_of_days = int(number_of_days)

st.markdown('<h3 align="center", style="color:darkolivegreen"> <u> Further Analysis </u> </h3> ', unsafe_allow_html = True)

st.markdown('<h6 align="left", style="color:lightpink"> Warning: Do not use any metric below in isolation (Refer additional details for help)!! </h6> ', unsafe_allow_html = True)

st.markdown(f"""<p align="left", style="color:dimgrey"> 
* {np.round(number_of_days,0)} Day Rolling Daily Cases CAGR <br>
* {np.round(number_of_days,0)} Day Rolling Deaths CAGR <br>
* Percentage change in Daily Deaths due to a 100% change in Daily Cases, based on data over the last {np.round(number_of_days,0)} Days (Beta(Slope)) <br>
* Reliability of Beta(Slope) given by R-squared 
</p> """,
 unsafe_allow_html = True)

# More details on measures
with st.expander("See Explanation on Beta (Slope) and R-squared"):

    st.write(f""" <p align = "left", style="color:lightblue">

    * The Beta (Slope) value can be interpreted as the percentage change (increase or decrease) in the number of daily deaths due to a percentage change in the number of daily cases. If the Beta (Slope) value is positive, then there is a positive relationship between the daily cases and daily deaths. Whilst if the Beta (Slope) value is negative then there is a negative relationship between the daily cases and daily deaths. <br>

    * The Rsquared measure can take values between 0 and 1 and is an indication on the reliability of the Beta (Slope) value computed above. If the value is closer to one, then the Beta (Slope) value is more reliable and if the value is closer to 0, then the Beta (Slope) value is less reliable. <br>

    * The Beta (Slope) metric is a dynamic one. This metric is a rolling one, computed using the daily cases and daily deaths over the last {number_of_days} days. The metric itself was computed by carrying out a linear regression between the log of daily deaths and the log of daily cases. <br> 

    * The Beta (Slope) metric cannot be interpreted in isolation. It must be studied in conjunction with the {number_of_days} Day Daily cases CAGR, {number_of_days} Day Daily Deaths CAGR, the number of Daily Cases and the number of Daily Deaths. <br>

    * In general, when the {number_of_days} Day Daily cases CAGR is rising faster than the {number_of_days} Day Daily Deaths CAGR, we will see that the Beta (Slope) value will dip. This is because the rate of increase in daily deaths is not as fast as the rate of increase in daily cases. Conversely if the {number_of_days} Day Daily Deaths CAGR is rising faster than the {number_of_days} Day Daily cases CAGR, then we will see an uptick in the Beta (Slope) value. This is because the rate of increase in daily deaths is faster than the rate of increase in daily cases. <br>

    <h6> Examples </h6> 

    * **Case 01:** Suppose that the Beta (Slope) value is 0.8 and that the daily cases in Country X is currently at 1000 and the daily deaths are at 100.  Then a Beta (Slope) value of 0.8 indicates that a 100% increase in daily cases (which is quite possible with a new variant) will result in 80 more people dying on a daily basis and this is a serious cause for concern. It would be even more concerning if the {number_of_days} Day Daily cases CAGR is remaining high and the {number_of_days} Day Daily Deaths CAGR is rising faster than the {number_of_days} Day Daily cases CAGR. <br> 

    * **Case 02:** Suppose that the Beta (Slope) value is 0.8 and that the daily cases in Country X is currently at 1000 and the daily deaths are at 10.  Then a Beta (Slope) value of 0.8 indicates that a 100% increase in daily cases (which is quite possible with a new variant) will result in 8 more people dying on a daily basis, and this might not be as serious as the case above. It would be even less concerning if the {number_of_days} Day Daily cases CAGR is trending down and the {number_of_days} Day Daily cases CAGR is higher than the {number_of_days} Day Daily Deaths CAGR.
    </p>""", unsafe_allow_html=True)


fig_3 = Plots_new.plot4(Global_cases, Global_deaths, Continent = continents_reverse[result_11], show_country=True_False_dict[result], Country = result_12,number_of_days = max(7,int(number_of_days)), corr_func = data_prep.corr, cagr_func=data_prep.cagr)

st.plotly_chart(fig_3)
