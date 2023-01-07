import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

import plotly.graph_objs as go
from ipywidgets import *

def InteractiveScatter(df,dforiginal):
    
    
    # Create FigureWidget with scatter plot of gdpPercap and lifeExp
    f = go.FigureWidget([go.Scatter(x=df['ABV'], y=df['review_overall'], mode='markers' ,text=dforiginal['Name'],customdata = dforiginal["Name"])])
    f.update_layout(title='ABV vs review_overall', xaxis=dict(
            title="ABV"
        ),
        yaxis=dict(
            title="review_overall"
        )
        )  
    #f.update_traces(hoverlabel=dforiginal['Name'].to_dict(), selector=dict(index=0))
    # Add linear regression to the plot
    f.add_shape(
        type='line',
        x0=min(df['ABV']),
        y0=df['ABV'].apply(lambda x: np.poly1d(np.polyfit(df['ABV'], df['review_overall'], 1))(x)).min(),
        x1=max(df['ABV']),
        y1=df['ABV'].apply(lambda x: np.poly1d(np.polyfit(df['ABV'], df['review_overall'], 1))(x)).max(),
        yref='y',
        line=dict(color='red')
    )

    # Define a function to update the scatter plot when a new x column is selected
    def update_plot(x , log):
        print(log)
        if log ==0:
            f.data[0].x = df[x]
            f.layout.shapes[0].y0 = df[x].apply(lambda a: np.poly1d(np.polyfit(df[x], df['review_overall'], 1))(a)).min()
            f.layout.shapes[0].y1 = df[x].apply(lambda a: np.poly1d(np.polyfit(df[x], df['review_overall'], 1))(a)).max()
            f.layout.shapes[0].x0 = min(df[x])
            f.layout.shapes[0].x1 = max(df[x])
        else :
            df_new = df.copy()
            #df_new[x] = df_new[x].apply(lambda x: np.log(x+1))
            df_new['review_overall'] = df_new['review_overall'].apply(lambda a: np.log(a+1))
            f.data[0].x = df_new[x]
            f.layout.shapes[0].y0 = df_new[x].apply(lambda a: np.poly1d(np.polyfit(df_new[x], df_new['review_overall'], 1))(a)).min()
            f.layout.shapes[0].y1 = df_new[x].apply(lambda a: np.poly1d(np.polyfit(df_new[x], df_new['review_overall'], 1))(a)).max()
            f.layout.shapes[0].x0 = min(df_new[x])
            f.layout.shapes[0].x1 = max(df_new[x])
        f.update_layout(title=f'{x} vs review_overall', xaxis=dict(
            title=f'{x}'
        ),
        yaxis=dict(
            title="review_overall"
        ) )
        # Update linear regression
        

    
    isinlog=0
      
    def update_plot_log(isinlog):
        isinlog = 0 if isinlog == 1 else 1
        update_plot(f.layout.xaxis.title.text,isinlog)
        
    
    
    
    logbutton = Button(description="Log")
    logbutton.on_click(lambda b: update_plot_log(isinlog))
    
    
    # Create buttons for each column to use as x values
    buttons = []
    for col in df.columns:
        button = Button(description=col)
        button.on_click(lambda b: update_plot(b.description , isinlog))
        buttons.append(button)

    # Divide buttons into two lines
    line1 = HBox(buttons[:int(len(buttons)/2)])
    line2 = HBox(buttons[int(len(buttons)/2):])
    display(VBox([f,logbutton, line1, line2]))

def InteractiveBoxPlot(df):
    # Load data and create a box plot with Plotly Express

    # Create FigureWidget with box plot of review_overall
    f = go.FigureWidget([go.Box(y=df['review_overall'])])
    f.update_layout(title='Box plot : review_overall')

    # Define a function to update the box plot when a new x column is selected
    def update_plot(x):
        f.data[0].y = df[x]
        f.update_layout(title=f'Box plot : {x}')

    # Create buttons for each column to use as x values
    buttons = []
    for col in df.columns:
        button = Button(description=col)
        button.on_click(lambda b: update_plot(b.description))
        buttons.append(button)

    # Divide buttons into two lines
    line1 = HBox(buttons[:int(len(buttons)/2)])
    line2 = HBox(buttons[int(len(buttons)/2):])

    # Display the FigureWidget and buttons
    display(VBox([f, line1, line2]))
