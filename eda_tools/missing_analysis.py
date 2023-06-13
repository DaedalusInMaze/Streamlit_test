import missingno
import pandas as pd
import numpy as np

## Specify exception values by user


def columns_to_analyze_missing(df, threshold = 0.015):
        '''
        Returns a list of columns to analyze missing values
        '''
        return df.loc[:, df.isnull().mean() > threshold].columns.tolist()



def missing_analysis_matrix(df, columns = None, threshold = 0.015, figsize = (10,10), fontsize = 12, color = (0.25, 0.25, 0.25)):
    '''
    Plots the missing values for the columns in the dataframe
    '''
    if columns is None:
        columns = columns_to_analyze_missing(df, threshold)
    missingno.matrix(df[columns], figsize = figsize, fontsize = fontsize, color = color)




def missing_analysis_bar(df, columns = None, threshold = 0.015, figsize = (10,10), fontsize = 12, color = (0.25, 0.25, 0.25)):
    '''
    Plots the missing values for the columns in the dataframe
    '''
    if columns is None:
        columns = columns_to_analyze_missing(df, threshold)
    missingno.bar(df[columns], figsize = figsize, fontsize = fontsize, color = color)



def missing_analysis_heatmap(df, columns = None, threshold = 0.015, figsize = (10,10), fontsize = 12, color = (0.25, 0.25, 0.25)):
    '''
    Plots the missing values for the columns in the dataframe
    '''
    if columns is None:
        columns = columns_to_analyze_missing(df, threshold)
    missingno.heatmap(df[columns], figsize = figsize, fontsize = fontsize, color = color)
    
    
def top_missing_variables(df, count = 50):
    '''
    List top n missing varaibles, and its missing count, missing percentage with two digits and percentage format, in descending order
    '''
    missing_count = df.isnull().sum().sort_values(ascending = False)
    missing_percentage = (df.isnull().mean() * 100).sort_values(ascending = False).round(2).astype(str) + '%'
    missing_df = pd.concat([missing_count, missing_percentage], axis = 1, keys = ['missing_count', 'missing_percentage'])
    return missing_df.head(count)   


def top_exception_variables(df, value: list, show_rows=30):
    '''
    Do Exception/Missing analysis with the user provided value, and print the top `show_rows` columns with the most exceptions
    '''
    try:
        exception_count = df.isin(value).sum().sort_values(ascending=False)
        exception_percentage = (df.isin(value).mean() * 100).sort_values(ascending=False).round(2).astype(str) + '%'
        exception_df = pd.concat([exception_count, exception_percentage], axis=1, keys=['exception_count', 'exception_percentage'])
        return exception_df.head(show_rows)
    except Exception as e:
        print("An error occurred:", e)
