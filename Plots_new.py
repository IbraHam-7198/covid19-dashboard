# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from scipy.stats import pearsonr
import pycountry_convert as pc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import data_prep
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

# Plots
continents = {'NA': 'North America','SA': 'South America', 'AS': 'Asia','OC': 'Oceania',
'AF': 'Africa','EU': 'Europe', 'World': 'World'}

# Placeholders
def placeholders(fig, row, col, place, show_annotations = 'Yes'):
    
    Annotations = ['Alpha', 'Beta', 'Gamma', 'Delta','Delta(2nd Wave)', 'Omicron','Vaccination Efforts begin']

    Dates = ['2020-12-18', '2020-12-18', '2021-01-11','2021-04-04','2021-08-10','2021-11-24','2020-12-01']

    colors = ['lightpink', 'lightpink','lightpink', 'lightpink','lightpink','salmon','navy']

    elevations = [0,1.25,1.8,1.8,1.8,1.8,2.5]

    for i,j in enumerate(Annotations): 
        
        if show_annotations == 'Yes':
            
            fig.add_vline(np.datetime64(Dates[i]), line = dict(color=colors[i], width = 1.5), row = row, col = col)

            fig.add_annotation(x = np.datetime64(Dates[i]), y = np.mean(place)*elevations[i], text = j, showarrow=False, 
                               bgcolor = colors[i], font = dict(color = 'white', size = 8),xanchor= 'center',row = row, col = col)
        else:

            fig.add_vline(np.datetime64(Dates[i]), line = dict(color=colors[i], width = 1.5), row = row, col = col)

            fig.add_annotation(x = np.datetime64(Dates[i]), y = np.mean(place)*elevations[i], text = '', showarrow=False, 
                               bgcolor = colors[i], font = dict(color = 'white', size = 8),xanchor= 'center',row = row, col = col)
                
    return fig

# Plot 01:

def plot1(cases, deaths, Select, type_of_graph):
    

    if Select == 'World':
        
        fig_1 = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]], subplot_titles = ['World'])

        y_val_1 = np.array(cases.groupby('Date').sum()[type_of_graph]).clip(0)

        y_val_2 = np.array(deaths.groupby('Date').sum()[type_of_graph]).clip(0)

    else:
        
        fig_1 = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]], subplot_titles = [continents[Select]])

        y_val_1 = np.array(cases[cases.Continent == Select].groupby('Date').sum()[type_of_graph]).clip(0)

        y_val_2 = np.array(deaths[deaths.Continent == Select].groupby('Date').sum()[type_of_graph]).clip(0)
        
    #Plot
    
    if type_of_graph == 'Agg_Daily':

        line_width = 3
        
    else:
        
        line_width = 2
        

    fig_1.add_trace(go.Scatter(x = np.unique(deaths.Date), 
                               y = y_val_2, 
                               mode = 'lines+markers', name = f'{type_of_graph} Deaths_' + Select, line = dict(color='greenyellow', width = line_width), legendgroup = 1,
                             marker = dict(color = 'navy', size = 1)),secondary_y = True)
    
    fig_1.add_trace(go.Scatter(x = np.unique(cases.Date), 
                               y = y_val_1, 
                               mode = 'lines+markers',name = f'{type_of_graph} Cases_' + Select, line = dict(color = 'powderblue', width = line_width), legendgroup = 1,
                         marker = dict(color = 'darkolivegreen', size = 2)),secondary_y = False)

    # Placeholders
    
    fig_1 = placeholders(fig = fig_1, row = 1, col = 1, place = y_val_1, show_annotations='Yes')
        

    fig_1.update_layout(showlegend = True, legend_font=dict(size = 8), width = 1030, height = 500, plot_bgcolor = 'white', 
                        hoverlabel_bgcolor = 'lightblue')
                        #title = {'text' :f'{type_of_graph} Cases and {type_of_graph} Deaths','xanchor': 'center', 'x':0.5},
                        #font = dict(color = 'navy', family = 'Arial', size = 10))

    fig_1.update_xaxes(showline = True, linecolor = 'lightgrey')

    fig_1.update_yaxes(showline = True, linecolor = 'lightgrey', secondary_y= True,
                       title_text = f'{type_of_graph}', color = 'silver', title_font_size = 8, 
                       range = [0,max(y_val_2)*1.5])

    fig_1.update_yaxes(showline = True, linecolor = 'lightgrey', secondary_y= False, 
                       title_text = f'{type_of_graph}', color = 'silver', title_font_size = 8,
                       range = [0,max(y_val_1)*1.05])
    
    return fig_1

# Plot 02:
def plot2(cases, deaths,Select,Country,type_of_graph):
    
    # Plot 01 Data

    if Select == 'World':
        
        fig_1 = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}],[{"secondary_y": True}]],vertical_spacing = 0.1,
                     subplot_titles = ['World', Country], shared_xaxes=True)

        y_val_1 = np.array(cases.groupby('Date').sum()[type_of_graph]).clip(0)

        y_val_2 = np.array(deaths.groupby('Date').sum()[type_of_graph]).clip(0)
        
        # Plot 02 Data

        y_val_11 = np.array(cases[cases.Country == Country].groupby('Date').sum()[type_of_graph]).clip(0)

        y_val_21 = np.array(deaths[deaths.Country == Country ].groupby('Date').sum()[type_of_graph]).clip(0)


    else:
        
        fig_1 = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}],[{"secondary_y": True}]],vertical_spacing = 0.1,
                     subplot_titles = [continents[Select], Country], shared_xaxes=True)
        
        # Plot 01 Data
        
        y_val_1 = np.array(cases[cases.Continent == Select].groupby('Date').sum()[type_of_graph]).clip(0)

        y_val_2 = np.array(deaths[deaths.Continent == Select].groupby('Date').sum()[type_of_graph]).clip(0)
        
        # Plot 02 Data
        
        y_val_11 = np.array(cases[(cases.Continent == Select) &(cases.Country == Country)].groupby('Date').sum()[type_of_graph]).clip(0)

        y_val_21 = np.array(deaths[(deaths.Continent == Select) & (deaths.Country == Country)].groupby('Date').sum()[type_of_graph]).clip(0)

    
    if type_of_graph == 'Agg_Daily':

        line_width_1 = 3
        
        line_width_2 = 3

    else:
        
        line_width_1 = 2
        
        line_width_2 = 2
        

    #Plot 1

    fig_1.add_trace(go.Scatter(x = np.unique(cases.Date), 
                               y = y_val_1, 
                               mode = 'lines+markers',name = f'{type_of_graph} Cases_' + Select, line = dict(color = 'greenyellow', width = line_width_1), legendgroup = 1,
                         marker = dict(color = 'navy', size = 1)),secondary_y = False,row = 1, col = 1)

    fig_1.add_trace(go.Scatter(x = np.unique(deaths.Date), 
                               y = y_val_2, 
                               mode = 'lines+markers', name = f'{type_of_graph} Deaths_' + Select, line = dict(color='powderblue', width =line_width_1/2*1.5), legendgroup = 1,
                             marker = dict(color = 'darkolivegreen', size = 1)),secondary_y = True,row = 1, col = 1)

    # Placeholders 

    fig_1 = placeholders(fig = fig_1, row = 1, col = 1, place = y_val_1, show_annotations='Yes')                            


    #Plot 2 
    
    fig_1.add_trace(go.Scatter(x = np.unique(cases.Date), 
                               y = y_val_11, 
                               mode = 'lines+markers',name = f'{type_of_graph} Cases_' + Country, line = dict(color = 'greenyellow', width = line_width_2), legendgroup = 2,
                         marker = dict(color = 'navy', size = 1)),secondary_y = False,row = 2, col = 1)

    fig_1.add_trace(go.Scatter(x = np.unique(deaths.Date), 
                               y = y_val_21, 
                               mode = 'lines+markers', name = f'{type_of_graph} Deaths_' + Country, line = dict(color='powderblue', width = line_width_2), legendgroup =2,
                             marker = dict(color = 'darkolivegreen', size = 1)),secondary_y = True,row = 2, col = 1)

    fig_1 = placeholders(fig = fig_1, row = 2, col = 1, place = y_val_11, show_annotations='No')                            

    fig_1.update_layout(showlegend = True, legend_font=dict(size = 8), width = 1030, height = 600, plot_bgcolor = 'white', 
                        hoverlabel_bgcolor = 'lightblue', legend_tracegroupgap = 180)
                        #title = {'text' :f'{type_of_graph} Cases and {type_of_graph} Deaths','xanchor': 'center', 'x':0.5},
                        #font = dict(color = 'navy', family = 'Arial', size = 10), title_pad_b = 2)

    fig_1.update_xaxes(showline = True, linecolor = 'lightgrey')

    fig_1.update_yaxes(showline = True, linecolor = 'lightgrey', secondary_y= True,
                       title_text = 'Deaths', color = 'silver', title_font_size = 8, row = 1, col = 1,
                       range = [0, max(y_val_2)*1.5])

    fig_1.update_yaxes(showline = True, linecolor = 'lightgrey', secondary_y= True,
                       title_text = 'Deaths', color = 'silver', title_font_size = 8, row = 2, col = 1,
                       range = [0, max(y_val_21)*1.5])

    fig_1.update_yaxes(showline = True, linecolor = 'lightgrey', secondary_y= False, 
                       title_text = 'Cases', color = 'silver', title_font_size = 8,
                       range = [0,max(y_val_1)*1.05], row = 1, col = 1)
 
    fig_1.update_yaxes(showline = True, linecolor = 'lightgrey', secondary_y= False, 
                       title_text = 'Cases', color = 'silver', title_font_size = 8,
                       range = [0,max(y_val_11)*1.05], row = 2, col = 1)

    return fig_1

# Function that plots multiple charts on demand 
def Cases_deaths(func1, func2, cases, deaths,Country,Select,type_of_graph, show_country = True):
    
    if show_country == True:
        
        fig = func2(cases = cases, deaths = deaths, Country = Country,Select = Select, type_of_graph = type_of_graph)
    
    else:
        
        fig = func1(cases = cases, deaths = deaths,Select = Select,type_of_graph = type_of_graph)
    
    return fig


# Plot 03
def plot3(cases, deaths,Continent, show_country, Country,number_of_days,corr_func = data_prep.corr, cagr_func = data_prep.cagr):
    
    fig = make_subplots(rows=1, cols=1)
    
    # Input data
    
    world = corr_func(cases, deaths, level = 'World', input_data=None, number_of_days=number_of_days)
    
    world_cagr = cagr_func(cases, level = 'World', input_data=None, number_of_days = number_of_days)
    
    continent = corr_func(cases, deaths, level = 'Continent', input_data=Continent, number_of_days=number_of_days)
    
    continent_cagr = cagr_func(cases, level = 'Continent', input_data=Continent, number_of_days = number_of_days)
    
    country = corr_func(cases, deaths, level = 'Country', input_data=Country, number_of_days=number_of_days)
    
    country_cagr = cagr_func(cases, level = 'Country', input_data=Country, number_of_days = number_of_days)
    
    
    if show_country == False:
        
        fig = make_subplots(rows=2, cols=1,specs=[[{"secondary_y": True}],[{"secondary_y": True}]], subplot_titles = ['World', continents[Continent]], shared_xaxes=True)
        
        # World
        
        
        
        fig.add_trace(go.Scatter(x = world['Date'], y = np.round(world['Correlation'],2).clip(-1), mode = 'lines+markers',
                         name = 'World_corr', line = dict(color='greenyellow', width = 1), legendgroup = 1,
                         marker = dict(color = 'navy', size = 0.5)), row = 1, col = 1, secondary_y = False)
        
        fig.add_trace(go.Scatter(x = world_cagr['Date'], y = np.round(world_cagr['CAGR'],2)*100, mode = 'lines+markers',
                         name = 'World_cagr', line = dict(color='hotpink', width = 0.9), legendgroup = 1,
                         marker = dict(color = 'yellowgreen', size = 0.5)), row = 1, col = 1, secondary_y = True) 

        fig.add_trace(go.Scatter(x = country_cagr['Date'], y = np.round(deaths.Daily/cases.Daily,4)*100, mode = 'lines+markers',
                         name = 'World Daily Deaths/Cases %', line = dict(color='lightblue', width = 0.9), legendgroup = 1,
                         marker = dict(color = 'yellowgreen', size = 0.5)), row = 1 , col = 1,secondary_y = True)   

        fig = placeholders(fig = fig, row = 1, col = 1, place = world['Correlation']*0.4, show_annotations='Yes')
        
        # Continent
                      
        fig.add_trace(go.Scatter(x = continent['Date'], y = np.round(continent['Correlation'],2).clip(-1), mode = 'lines+markers',
                 name = f'{Continent}_corr', line = dict(color='greenyellow', width = 1), legendgroup = 2,
                 marker = dict(color = 'hotpink', size = 0.5)),row = 2, col = 1, secondary_y = False)
        

        fig.add_trace(go.Scatter(x = continent_cagr['Date'], y = np.round(continent_cagr['CAGR'],2)*100, mode = 'lines+markers',
                 name = f'{Continent}_cagr', line = dict(color='hotpink', width = 0.9),legendgroup = 2, 
                 marker = dict(color = 'yellowgreen', size = 0.5)),row = 2, col = 1, secondary_y = True)

        fig.add_trace(go.Scatter(x = country_cagr['Date'], y = np.round(deaths[deaths.continent == Continent].Daily/cases[cases.continent == Continent].Daily,4)*100, mode = 'lines+markers',
                         name = f'{Continent}_Daily Deaths/Cases %', line = dict(color='lightblue', width = 0.9), legendgroup = 2,
                         marker = dict(color = 'yellowgreen', size = 0.5)), row = 2 , col = 1,secondary_y = True)          

        fig = placeholders(fig = fig, row = 2, col = 1, place = world['Correlation']*0.4, show_annotations='No')
        
        fig.update_layout(showlegend = True, legend_font=dict(size = 8), width = 1000, height = 650, plot_bgcolor = 'white', 
                            hoverlabel_bgcolor = 'lightblue', legend_tracegroupgap = 160) 
                           # title = {'text' :f'{number_of_days} Day Moving Correlation between Daily Cases and Daily Deaths + #{number_of_days} Day CAGR of Daily Cases ','xanchor': 'center', 'x':0.5},
                           # font = dict(color = 'darkgreen', family = 'Arial',size = 10))
    else:
        
        fig = make_subplots(rows=3, cols=1,subplot_titles = ['World',continents[Continent],Country],vertical_spacing = 0.1,
                           specs=[[{"secondary_y": True}],[{"secondary_y": True}],[{"secondary_y": True}]], shared_xaxes=True)
        
        # World
        
        fig.add_trace(go.Scatter(x = world['Date'], y = np.round(world['Correlation'],2).clip(-1), mode = 'lines+markers',
                         name = 'World_corr', line = dict(color='greenyellow', width = 1),legendgroup = 1, 
                         marker = dict(color = 'yellowgreen', size = 0.5)), row = 1, col = 1,secondary_y = False)
        
        fig.add_trace(go.Scatter(x = world_cagr['Date'], y = np.round(world_cagr['CAGR'],2)*100, mode = 'lines+markers',
                 name = 'World_cagr', line = dict(color='hotpink', width = 0.9), legendgroup = 1,
                 marker = dict(color = 'yellowgreen', size = 0.5)), row = 1, col = 1,secondary_y = True) 

        fig.add_trace(go.Scatter(x = country_cagr['Date'], y = np.round(deaths.Daily/cases.Daily,4)*100, mode = 'lines+markers',
                         name = 'World Daily Deaths/Cases %', line = dict(color='lightblue', width = 0.9), legendgroup = 1,
                         marker = dict(color = 'yellowgreen', size = 0.5)), row = 1 , col = 1,secondary_y = True)   

        fig = placeholders(fig = fig, row = 1, col = 1, place = world['Correlation']*0.4)
        
        
        # Continent
        
               
        fig.add_trace(go.Scatter(x = continent['Date'], y = np.round(continent['Correlation'],2).clip(-1), mode = 'lines+markers',
            name = f'{Continent}_corr', line = dict(color='greenyellow', width = 1), legendgroup = 2,
            marker = dict(color = 'yellowgreen', size = 1)), row = 2, col = 1,secondary_y = False)
        
        fig.add_trace(go.Scatter(x = continent_cagr['Date'], y = np.round(continent_cagr['CAGR'],2)*100, mode = 'lines+markers',
                                 name = f'{Continent}_cagr', line = dict(color='hotpink', width = 0.9), legendgroup = 2,
                                 marker = dict(color = 'yellowgreen', size = 0.5)),row = 2, col = 1, secondary_y = True)

        fig.add_trace(go.Scatter(x = country_cagr['Date'], y = np.round(deaths[deaths.Continent == Continent].Daily/cases[cases.Continent == Continent].Daily,4)*100, mode = 'lines+markers',
                         name = f'{Continent}_Daily Deaths/Cases %', line = dict(color='lightblue', width = 0.9), legendgroup = 2,
                         marker = dict(color = 'yellowgreen', size = 0.5)), row = 2 , col = 1,secondary_y = True)          

        fig = placeholders(fig = fig, row = 2, col = 1, place = world['Correlation']*0.4, show_annotations='No')

        
        #Country
        
        
        fig.add_trace(go.Scatter(x = country['Date'], y = np.round(country['Correlation'],2).clip(-1), mode = 'lines+markers',
             name = f'{Country}_corr', line = dict(color='greenyellow', width = 1), legendgroup = 3,
             marker = dict(color = 'yellowgreen', size = 1)),row = 3, col = 1,secondary_y = False)

        fig.add_trace(go.Scatter(x = country_cagr['Date'], y = np.round(country_cagr['CAGR'],2)*100, mode = 'lines+markers',
                         name = f'{Country}_cagr', line = dict(color='hotpink', width = 0.9), legendgroup = 3,
                         marker = dict(color = 'yellowgreen', size = 0.5)), row = 3 , col = 1,secondary_y = True)

        fig.add_trace(go.Scatter(x = country_cagr['Date'], y = np.round(deaths[deaths.Country == Country].Daily/cases[cases.Country == Country].Daily,4)*100, mode = 'lines+markers',
                         name = f'{Country}_Daily Deaths/Cases %', line = dict(color='lightblue', width = 0.9), legendgroup = 3,
                         marker = dict(color = 'yellowgreen', size = 0.5)), row = 3 , col = 1,secondary_y = True)    

                           

        fig = placeholders(fig = fig, row = 3, col = 1, place = world['Correlation']*0.4, show_annotations='No')     
        
        fig.update_layout(showlegend = True, legend_font=dict(size = 8), width = 1000, height = 750, plot_bgcolor = 'white', 
                            hoverlabel_bgcolor = 'lightblue',legend_tracegroupgap = 160) 
                            #title = {'text' :f'{number_of_days} Day Moving Correlation between Daily Cases and Daily Deaths + #{number_of_days} Day CAGR of Daily Cases ','xanchor': 'center', 'x':0.5},
                            #font = dict(color = 'darkgreen', family = 'Arial', size = 10))
    
    fig.add_hline(y = 0, line_color = 'grey', line_width = 2, opacity = 0.5)  
    
    fig.update_xaxes(showline = True, linecolor = 'grey')

    fig.update_yaxes(showline = True, linecolor = 'lightgrey', secondary_y= True,
                       title_text = 'CAGR + Daily Deaths/ Cases %', color = 'silver', title_font_size = 8,range = [-100,100])

    fig.update_yaxes(showline = True, linecolor = 'lightgrey', secondary_y= False, 
                       title_text = 'Correlation', color = 'silver', title_font_size = 8, range = [-1,1])
    
    return fig

# Plot 04
def plot4(cases, deaths,Continent, show_country, Country,number_of_days,corr_func = data_prep.corr, cagr_func = data_prep.cagr):
    
    # Input data

    # World data

    world = corr_func(cases, deaths, level = 'World', input_data=None, number_of_days=number_of_days)

    world_cagr = cagr_func(cases, level = 'World', input_data=None, number_of_days = number_of_days)

    world_cagr_d = cagr_func(deaths, level = 'World', input_data=None, number_of_days = number_of_days) 

    world_coeff = RollingOLS(exog = np.log(cases.groupby('Date').sum()['Daily']+0.1).tolist(),
    endog = np.log(deaths.groupby('Date').sum()['Daily']+0.1).tolist(), window=number_of_days).fit().params.ravel()

    world_r_2 = RollingOLS(exog = np.log(cases.groupby('Date').sum()['Daily']+0.1).tolist(),
    endog = np.log(deaths.groupby('Date').sum()['Daily']+0.1).tolist(), window=number_of_days).fit().rsquared

    # Country data

    if show_country == True:

        country_cagr = cagr_func(cases, level = 'Country', input_data=Country, number_of_days = number_of_days)

        country_cagr_d = cagr_func(deaths, level = 'Country', input_data=Country, number_of_days = number_of_days)

        country_coeff = RollingOLS(exog = np.log(cases[cases.Country == Country].groupby('Date').sum()['Daily']+0.1).tolist(), endog = np.log(deaths[deaths.Country == Country].groupby('Date').sum()['Daily']+0.1).tolist(), window=number_of_days).fit().params.ravel()

        country_r_2 = RollingOLS(exog = np.log(cases[cases.Country == Country].groupby('Date').sum()['Daily']+0.1).tolist(),endog = np.log(deaths[deaths.Country == Country].groupby('Date').sum()['Daily']+0.1).tolist(), window=number_of_days).fit().rsquared
    
    else:
        pass

    # Continent data (When continent is not world)

    if Continent != 'World':
   
        continent_cagr = cagr_func(cases, level = 'Continent', input_data=Continent, number_of_days = number_of_days)

        continent_cagr_d = cagr_func(deaths, level = 'Continent', input_data=Continent, number_of_days = number_of_days)

        continent_coeff = RollingOLS(exog = np.log(cases[cases.Continent == Continent].groupby('Date').sum()['Daily']+0.1).tolist(),endog = np.log(deaths[deaths.Continent == Continent].groupby('Date').sum()['Daily']+0.1).tolist(), window=number_of_days).fit().params.ravel()

        continent_r_2 = RollingOLS(exog = np.log(cases[cases.Continent == Continent].groupby('Date').sum()['Daily']+0.1).tolist(), endog = np.log(deaths[deaths.Continent == Continent].groupby('Date').sum()['Daily']+0.1).tolist(), window=number_of_days).fit().rsquared

        # Country data

        if show_country == True:

            country_cagr = cagr_func(cases, level = 'Country', input_data=Country, number_of_days = number_of_days)

            country_cagr_d = cagr_func(deaths, level = 'Country', input_data=Country, number_of_days = number_of_days)

            country_coeff = RollingOLS(exog = np.log(cases[cases.Country == Country].groupby('Date').sum()['Daily']+0.1).tolist(), endog = np.log(deaths[deaths.Continent == Continent].groupby('Date').sum()['Daily']+0.1).tolist(), window=number_of_days).fit().params.ravel()

            country_r_2 = RollingOLS(exog = np.log(cases[cases.Country == Country].groupby('Date').sum()['Daily']+0.1).tolist(),endog = np.log(deaths[deaths.Country == Country].groupby('Date').sum()['Daily']+0.1).tolist(), window=number_of_days).fit().rsquared
    
    else:

        pass

    if Continent != 'World':

        if show_country == False:
            
            fig = make_subplots(rows=2, cols=1,specs=[[{"secondary_y": True}],[{"secondary_y": True}]], subplot_titles = ['World', continents[Continent]], shared_xaxes=True)
            
            # World
                
            fig.add_trace(go.Scatter(x = world_cagr['Date'], y = np.round(world_cagr['CAGR'],2)*100, mode = 'lines+markers',
                            name = 'Cases_cagr', line = dict(color='greenyellow', width = 2), legendgroup = 1,
                            marker = dict(color = 'navy', size = 0.5)), row = 1, col = 1, secondary_y = True) 

            fig.add_trace(go.Scatter(x = world_cagr_d['Date'], y = np.round(world_cagr_d['CAGR'],2)*100, mode = 'lines+markers',
                            name = 'Deaths_cagr', line = dict(color='powderblue', width = 1.5), legendgroup = 1,
                            marker = dict(color = 'darkolivegreen', size = 0.5)), row = 1, col = 1, secondary_y = True) 

            fig.add_trace(go.Scatter(x = world_cagr_d['Date'], y = np.round(world_coeff,2), mode = 'lines+markers',
                            name = 'Beta(Slope)', line = dict(color='deepskyblue', width = 1.5), legendgroup = 1,
                            marker = dict(color = 'violet', size = 0.2)), row = 1, col = 1, secondary_y = False)

            fig.add_trace(go.Scatter(x = world_cagr_d['Date'], y = np.round(world_r_2,4), mode = 'lines+markers',
                            name = 'Rsquared', line = dict(color='thistle', width = 1.5), legendgroup = 1,
                            marker = dict(color = 'thistle', size = 0.5)), row = 1, col = 1, secondary_y = False)

            fig = placeholders(fig = fig, row = 1, col = 1, place = world['Correlation']*0.4, show_annotations='Yes')
            
            # Continent
                        
            fig.add_trace(go.Scatter(x = continent_cagr['Date'], y = np.round(continent_cagr['CAGR'],2)*100, mode = 'lines+markers',
                    name = 'Cases_cagr', line = dict(color='greenyellow', width = 2),legendgroup = 2, 
                    marker = dict(color = 'navy', size = 0.5)),row = 2, col = 1, secondary_y = True)

            fig.add_trace(go.Scatter(x = continent_cagr_d['Date'], y = np.round(continent_cagr_d['CAGR'],2)*100, mode = 'lines+markers',
                    name = 'Deaths_cagr', line = dict(color='powderblue', width = 1.5), legendgroup = 2,
                    marker = dict(color = 'darkolivegreen', size = 0.5)),row = 2, col = 1, secondary_y = True)

            fig.add_trace(go.Scatter(x = world_cagr_d['Date'], y = np.round(continent_coeff,2), mode = 'lines+markers',name = 'Beta(Slope)', line = dict(color='deepskyblue', width = 1.5), legendgroup = 2,
            marker = dict(color = 'violet', size = 0.2)), row = 2, col = 1, secondary_y = False)

            fig.add_trace(go.Scatter(x = world_cagr_d['Date'], y = np.round(continent_r_2,4), mode = 'lines+markers',
            name = 'Rsquared', line = dict(color='thistle', width = 1.5), legendgroup = 2,
            marker = dict(color = 'thistle', size = 0.5)), row = 2, col = 1, secondary_y = False)

            fig = placeholders(fig = fig, row = 2, col = 1, place = world['Correlation']*0.4, show_annotations='No')
            
            fig.update_layout(showlegend = True, legend_font=dict(size = 8), width = 1000, height = 650, plot_bgcolor = 'white', 
                                hoverlabel_bgcolor = 'lightblue', legend_tracegroupgap = 140) 
                            # title = {'text' :f'{number_of_days} Day Moving Correlation between Daily Cases and Daily Deaths + #{number_of_days} Day CAGR of Daily Cases ','xanchor': 'center', 'x':0.5},
                            # font = dict(color = 'darkgreen', family = 'Arial',size = 10))
        else:
            
            fig = make_subplots(rows=3, cols=1,subplot_titles = ['World',continents[Continent],Country],vertical_spacing = 0.1,
                            specs=[[{"secondary_y": True}],[{"secondary_y": True}],[{"secondary_y": True}]], shared_xaxes=True)
            
            # World
    
        
            fig.add_trace(go.Scatter(x = world_cagr['Date'], y = np.round(world_cagr['CAGR'],2)*100, mode = 'lines+markers',
                    name = 'Cases_cagr', line = dict(color='greenyellow', width = 2), legendgroup = 1,
                    marker = dict(color = 'navy', size = 0.5)), row = 1, col = 1,secondary_y = True) 

            fig.add_trace(go.Scatter(x = world_cagr_d['Date'], y = np.round(world_cagr_d['CAGR'],2)*100, mode = 'lines+markers',
                    name = 'Deaths_cagr', line = dict(color='powderblue', width = 1.5), legendgroup = 1,
                    marker = dict(color = 'darkolivegreen', size = 0.5)), row = 1, col = 1,secondary_y = True) 

            fig.add_trace(go.Scatter(x = world_cagr_d['Date'], y = np.round(world_coeff,2), mode = 'lines+markers',
                            name = 'Beta(Slope)', line = dict(color='deepskyblue', width = 1.5), legendgroup = 1,
                            marker = dict(color = 'violet', size = 0.2)), row = 1, col = 1, secondary_y = False)

            fig.add_trace(go.Scatter(x = world_cagr_d['Date'], y = np.round(world_r_2,4), mode = 'lines+markers',
                            name = 'Rsquared', line = dict(color='thistle', width = 1.5), legendgroup = 1,
                            marker = dict(color = 'thistle', size = 0.5)), row = 1, col = 1, secondary_y = False)

            fig = placeholders(fig = fig, row = 1, col = 1, place = world['Correlation']*0.4)
            
            
            # Continent
            
            fig.add_trace(go.Scatter(x = continent_cagr['Date'], y = np.round(continent_cagr['CAGR'],2)*100, mode = 'lines+markers',
                                    name = 'Cases_cagr', line = dict(color='greenyellow', width = 2), legendgroup = 2,
                                    marker = dict(color = 'navy', size = 0.5)),row = 2, col = 1, secondary_y = True)

            fig.add_trace(go.Scatter(x = continent_cagr_d['Date'], y = np.round(continent_cagr_d['CAGR'],2)*100, mode = 'lines+markers', name = 'Deaths_cagr', line = dict(color='powderblue', width = 1.5), legendgroup = 2,
                                    marker = dict(color = 'darkolivegreen', size = 0.5)),row = 2, col = 1, secondary_y = True)
    
            fig.add_trace(go.Scatter(x = world_cagr_d['Date'], y = np.round(continent_coeff,2), mode = 'lines+markers',name = 'Beta(Slope)', line = dict(color='deepskyblue', width = 1.5), legendgroup = 2,
            marker = dict(color = 'violet', size = 0.2)), row = 2, col = 1, secondary_y = False)

            fig.add_trace(go.Scatter(x = world_cagr_d['Date'], y = np.round(continent_r_2,4), mode = 'lines+markers',
            name = 'Rsquared', line = dict(color='thistle', width = 1.5), legendgroup = 2,
            marker = dict(color = 'thistle', size = 0.5)), row = 2, col = 1, secondary_y = False)

            fig = placeholders(fig = fig, row = 2, col = 1, place = world['Correlation']*0.4, show_annotations='No')

            
            #Country
            
            
            fig.add_trace(go.Scatter(x = country_cagr['Date'], y = np.round(country_cagr['CAGR'],2)*100, mode = 'lines+markers',
                            name = 'Cases_cagr', line = dict(color='greenyellow', width = 2), legendgroup = 3,
                            marker = dict(color = 'navy', size = 0.5)), row = 3 , col = 1,secondary_y = True)

            fig.add_trace(go.Scatter(x = country_cagr_d['Date'], y = np.round(country_cagr_d['CAGR'],2)*100, mode = 'lines+markers',
                            name = 'Deaths_cagr', line = dict(color='powderblue', width = 1.5), legendgroup = 3,
                            marker = dict(color = 'darkolivegreen', size = 0.5)), row = 3 , col = 1,secondary_y = True)

            fig.add_trace(go.Scatter(x = world_cagr_d['Date'], y = np.round(country_coeff,2), mode = 'lines+markers',name = 'Beta(Slope)', line = dict(color='deepskyblue', width = 1.5), legendgroup = 3,
            marker = dict(color = 'violet', size = 0.2)), row =3, col = 1, secondary_y = False)

            fig.add_trace(go.Scatter(x = world_cagr_d['Date'], y = np.round(country_r_2,4), mode = 'lines+markers',
            name = 'Rsquared', line = dict(color='thistle', width = 1.5), legendgroup = 3,
            marker = dict(color = 'thistle', size = 0.5)), row = 3, col = 1, secondary_y = False)
                            
            fig = placeholders(fig = fig, row = 3, col = 1, place = world['Correlation']*0.4, show_annotations='No')     
            
            fig.update_layout(showlegend = True, legend_font=dict(size = 8), width = 1000, height = 750, plot_bgcolor = 'white', 
                                hoverlabel_bgcolor = 'lightblue',legend_tracegroupgap = 140) 
                                #title = {'text' :f'{number_of_days} Day Moving Correlation between Daily Cases and Daily Deaths + #{number_of_days} Day CAGR of Daily Cases ','xanchor': 'center', 'x':0.5},
                                #font = dict(color = 'darkgreen', family = 'Arial', size = 10))
    

    else:

        if show_country == False:
            
            fig = make_subplots(rows=1, cols=1,specs=[[{"secondary_y": True}]], subplot_titles = ['World'], shared_xaxes=True)

            # World
                
            fig.add_trace(go.Scatter(x = world_cagr['Date'], y = np.round(world_cagr['CAGR'],2)*100, mode = 'lines+markers',
                            name = 'Cases_cagr', line = dict(color='greenyellow', width = 2), legendgroup = 1,
                            marker = dict(color = 'navy', size = 0.5)), row = 1, col = 1, secondary_y = True) 

            fig.add_trace(go.Scatter(x = world_cagr_d['Date'], y = np.round(world_cagr_d['CAGR'],2)*100, mode = 'lines+markers',
                            name = 'Deaths_cagr', line = dict(color='powderblue', width = 1.5), legendgroup = 1,
                            marker = dict(color = 'darkolivegreen', size = 0.5)), row = 1, col = 1, secondary_y = True) 

            fig.add_trace(go.Scatter(x = world_cagr_d['Date'], y = np.round(world_coeff,2), mode = 'lines+markers',
                            name = 'Beta(Slope)', line = dict(color='deepskyblue', width = 1.5), legendgroup = 1,
                            marker = dict(color = 'violet', size = 0.2)), row = 1, col = 1, secondary_y = False)

            fig.add_trace(go.Scatter(x = world_cagr_d['Date'], y = np.round(world_r_2,4), mode = 'lines+markers',
                            name = 'Rsquared', line = dict(color='thistle', width = 1.5), legendgroup = 1,
                            marker = dict(color = 'thistle', size = 0.5)), row = 1, col = 1, secondary_y = False)

            fig = placeholders(fig = fig, row = 1, col = 1, place = world['Correlation']*0.4, show_annotations='Yes')

            fig.update_layout(showlegend = True, legend_font=dict(size = 8), width = 1000, height = 400, plot_bgcolor = 'white', 
                                hoverlabel_bgcolor = 'lightblue',legend_tracegroupgap = 140) 
                                #title = {'text' :f'{number_of_days} Day Moving Correlation between Daily Cases and Daily Deaths + #{number_of_days} Day CAGR of Daily Cases ','xanchor': 'center', 'x':0.5},
                                #font = dict(color = 'darkgreen', family = 'Arial', size = 10))

        else:
            
            fig = make_subplots(rows=2, cols=1,subplot_titles = ['World',Country],vertical_spacing = 0.1,
                            specs=[[{"secondary_y": True}],[{"secondary_y": True}]], shared_xaxes=True)
            
            # World
         
            fig.add_trace(go.Scatter(x = world_cagr['Date'], y = np.round(world_cagr['CAGR'],2)*100, mode = 'lines+markers',
                    name = 'Cases_cagr', line = dict(color='greenyellow', width = 2), legendgroup = 1,
                    marker = dict(color = 'navy', size = 0.5)), row = 1, col = 1,secondary_y = True) 

            fig.add_trace(go.Scatter(x = world_cagr_d['Date'], y = np.round(world_cagr_d['CAGR'],2)*100, mode = 'lines+markers',
                    name = 'Deaths_cagr', line = dict(color='powderblue', width = 1.5), legendgroup = 1,
                    marker = dict(color = 'darkolivegreen', size = 0.5)), row = 1, col = 1,secondary_y = True) 

            fig.add_trace(go.Scatter(x = world_cagr_d['Date'], y = np.round(world_coeff,2), mode = 'lines+markers',
                            name = 'Beta(Slope)', line = dict(color='deepskyblue', width = 1.5), legendgroup = 1,
                            marker = dict(color = 'violet', size = 0.2)), row = 1, col = 1, secondary_y = False)

            fig.add_trace(go.Scatter(x = world_cagr_d['Date'], y = np.round(world_r_2,4), mode = 'lines+markers',
                            name = 'Rsquared', line = dict(color='thistle', width = 1.5), legendgroup = 1,
                            marker = dict(color = 'thistle', size = 0.5)), row = 1, col = 1, secondary_y = False)

            fig = placeholders(fig = fig, row = 1, col = 1, place = world['Correlation']*0.4)
            
            #Country
                    
            fig.add_trace(go.Scatter(x = country_cagr['Date'], y = np.round(country_cagr['CAGR'],2)*100, mode = 'lines+markers',
                            name = 'Cases_cagr', line = dict(color='greenyellow', width = 2), legendgroup = 3,
                            marker = dict(color = 'navy', size = 0.5)), row = 2 , col = 1,secondary_y = True)

            fig.add_trace(go.Scatter(x = country_cagr_d['Date'], y = np.round(country_cagr_d['CAGR'],2)*100, mode = 'lines+markers',
                            name = 'Deaths_cagr', line = dict(color='powderblue', width = 1.5), legendgroup = 3,
                            marker = dict(color = 'darkolivegreen', size = 0.5)), row = 2 , col = 1,secondary_y = True)

            fig.add_trace(go.Scatter(x = world_cagr_d['Date'], y = np.round(country_coeff,2), mode = 'lines+markers',name = 'Beta(Slope)', line = dict(color='deepskyblue', width = 1.5), legendgroup = 3,
            marker = dict(color = 'violet', size = 0.2)), row =2, col = 1, secondary_y = False)

            fig.add_trace(go.Scatter(x = world_cagr_d['Date'], y = np.round(country_r_2,4), mode = 'lines+markers',
            name = 'Rsquared', line = dict(color='thistle', width = 1.5), legendgroup = 3,
            marker = dict(color = 'thistle', size = 0.5)), row = 2, col = 1, secondary_y = False)
                            
            fig = placeholders(fig = fig, row = 2, col = 1, place = world['Correlation']*0.4, show_annotations='No')     
            
            fig.update_layout(showlegend = True, legend_font=dict(size = 8), width = 1000, height = 700, plot_bgcolor = 'white', 
                                hoverlabel_bgcolor = 'lightblue',legend_tracegroupgap = 200) 
                                #title = {'text' :f'{number_of_days} Day Moving Correlation between Daily Cases and Daily Deaths + #{number_of_days} Day CAGR of Daily Cases ','xanchor': 'center', 'x':0.5},
                                #font = dict(color = 'darkgreen', family = 'Arial', size = 10))
    
    fig.add_hline(y = 0, line_color = 'lightgrey', line_width = 2, opacity = 0.5,secondary_y= True)  
    
    fig.update_xaxes(showline = True, linecolor = 'grey')

    fig.update_yaxes(showline = True, linecolor = 'lightgrey', secondary_y= True,
                       title_text = 'CAGR + Daily Deaths/ Cases %', color = 'silver', title_font_size = 6, range = (-50,50), dtick = 10)

    fig.update_yaxes(showline = True, linecolor = 'lightgrey', secondary_y= False, 
                       title_text = 'Beta (Slope) + Rsquared', color = 'silver', title_font_size = 6)
    
    return fig   
