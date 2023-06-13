import pandas as pd
import numpy as np
import ydata_profiling as ydp


def hit_rate(vendor_df, perf_df, left_key, right_key):
    '''
    Caculate the hit rate of vendor dataset
    '''

    # Merge two dataset
    df = pd.merge(vendor_df, perf_df, left_on=left_key, right_on=right_key, how='inner')

    return df.shape[0] / perf_df.shape[0]


def bad_rate(df, performance):
    '''
    Get the bad rate (performance rate)
    '''

    return df[df[performance] == 1].shape[0]/df.shape[0]


class Correlation():
    def __init__(self, df, method, threshold = 0.6, columns = None):
        self.df = df
        self.columns = columns
        self.method = method
        self.corr = None
        self.threshold = threshold
        self.high_corr = None

    def correlation_table(self):
        '''
        Get the correlation table
        '''

        try:
            if self.columns == None:
                self.columns = self.df.columns

            methods = ["Pearson", "Spearman", "Kendall"]

            if self.method not in methods:
                raise ValueError("Invalid correlation method. Expected one of: %s" % methods)
                
            if self.method == 'Pearson':
                self.corr = self.df[self.columns].corr(method='pearson')
            elif self.method == 'Spearman':
                self.corr = self.df[self.columns].corr(method='spearman')
            elif self.method == 'Kendall':
                self.corr = self.df[self.columns].corr(method='kendall')

            self.corr = self.corr.round(2)
            self.corr = self.corr.mask(np.triu(np.ones(self.corr.shape)).astype(np.bool))

        except ValueError as ve:
            print(ve)
        except Exception as e:
            print("An unexpected error occurred:", e)
        else:
            return self.corr


    def get_highly_correlated_pairs(self):
        '''
        Get correlated variable paris whose absolute value is above the threshold
        '''
        high_corr = self.corr.unstack().dropna()
        abs_corr = high_corr.abs()
        self.high_corr = high_corr[abs_corr >= self.threshold]
        self.high_corr = self.high_corr.sort_values(ascending=False)
        self.high_corr = self.high_corr.reset_index()
        self.high_corr.columns = ['Variable 1', 'Variable 2', 'Correlation']

        return self.high_corr


    def get_directly_correlated_pairs(self):
        '''
        Get the variables that directly correlated, correlation = 1 or -1
        '''
        pairs = self.get_highly_correlated_pairs()
        direct_subset = pairs[pairs['Correlation'].isin([1, -1])]

        return direct_subset
    

    def get_highly_correlated_variables(self):
        '''
        Get the variables that are highly correlated
        '''
        pairs = self.get_highly_correlated_pairs()

        # Get variables that are highly correlated, which equals union of the two column minus the directly correlated variables
        subset = pairs[~pairs['Correlation'].isin([1, -1])]
        subset_var = subset['Variable 1'].tolist() + subset['Variable 2'].tolist()
        subset_var = list(set(subset_var))
        
        return subset_var
    

    def plot_heat_map(self, input = None, input_attributes = None):
        '''
        If the correlation paris are small, plot the heatmap
        '''
        if input is None:
            data = self.df
        else:
            data = input
        
        if input_attributes == None:
            attrs = self.get_highly_correlated_variables()
        else:
            attrs = input_attributes

        corr_profile = ydp.ProfileReport(data[attrs], config_file="profile_config/yprofile_correlation.yaml")
        corr_profile.config.interactions.targets = []
        corr_profile.to_notebook_iframe()
