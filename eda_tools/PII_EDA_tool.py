import pandas as pd
import numpy as np
import re
import plotly.express as px

class PII_EDA():

    def __init__(self, input_path):
        '''
        Initialize the class
        '''
        try:
            self.df = pd.read_csv(input_path, low_memory=False)
            self.pii = None
            self.pii_flag = None

        except FileNotFoundError:
            print("File not found.")

        except Exception as e:
            print("An error occurred:", e)


    def get_df(self):
        '''
        Return the dataframe
        '''
        return self.df
    

    def get_PII_data(self):
        '''
        Adjust column names
        '''
        self.pii = self.df[['p_inpclnaddrfull', 'p_inpclnphonehome', 'p_inpclnssn', 'p_inpclnnamefirst', 'p_inpclnnamelast']]
        self.pii = self.pii.rename(columns={
                'p_inpclnaddrfull': 'Address',
                'p_inpclnphonehome': 'Phone',
                'p_inpclnssn': 'SSN',
                'p_inpclnnamefirst': 'First Name',
                'p_inpclnnamelast': 'Last Name'
            })
        
        return self.pii
    
    
    def get_PII_flags(self):
        '''
        Get the columns with PII flags and rename
        '''
        self.pii_flag = self.df[['p_inpclnnamefirstflag', 'p_inpclnnamelastflag', 'p_inpclnaddrfullflag', 'p_inpclnphonehomeflag', 'p_inpclnssnflag', 'p_inpclndobflag']]
        self.pii_flag = self.pii_flag.rename(columns={
                'p_inpclnaddrfullflag': 'Address',
                'p_inpclnphonehomeflag': 'Phone',
                'p_inpclnssnflag': 'SSN',
                'p_inpclndobflag': 'DOB',
                'p_inpclnnamefirstflag': 'First Name',
                'p_inpclnnamelastflag': 'Last Name'
            })
        
        return self.pii_flag
    

    def identify_duplicates(self):
        '''
        Identify duplicated PII info
        '''
        # Create duplication table
        ds = self.get_PII_data()

        dupes = ds.assign(
            Name = lambda x: x['First Name'] + x['Last Name'],
            **{
                'Duplicated Address': lambda x: (x['Address'] != -99999) & (x['Address'] != -99998) & (x['Address'].duplicated()),
                'Duplicated Phone': lambda x: (x['Phone'] != -99999) & (x['Phone'] != -99998) & x['Phone'].duplicated(),
                'Duplicated SSN': lambda x: (x['SSN'] != -99999) & (x['SSN'] != -99998) & x['SSN'].duplicated(),
                'Duplicated Address + Name': lambda x: x['Duplicated Address'] & x.duplicated(subset=['Name']),
                'Duplicated Phone + Name': lambda x: x['Duplicated Phone'] & x.duplicated(subset=['Name']),
                'Duplicated SSN + Name': lambda x: x['Duplicated SSN'] & x.duplicated(subset=['Name'])
            }
        )

        return dupes


    def get_hit_rates(self):
        '''
        Get hit rates for each PII info
        '''    
        ds = self.get_PII_flags()
        ds_long = ds.stack().reset_index()
        ds_long.columns = ['Index', 'Input', 'value']
        ds_long.drop('Index', axis=1, inplace=True)
        ds_raw = ds_long[ds_long['value'] != -99999]
        ds_clean = ds_long[ds_long['value'] == 1]

        raw_count = ds_raw.groupby('Input').count()
        clean_count = ds_clean.groupby('Input').count()
        raw_hit_rate = raw_count / ds.shape[0]
        clean_hit_rate = clean_count / ds.shape[0]

        # make a table for the four columns
        hit_rate = pd.concat([raw_count, raw_hit_rate, clean_count, clean_hit_rate], axis=1)
        hit_rate.columns = ['Raw Count', 'Raw Hit Rate', 'Clean Count', 'Cleaned Hit Rate']

        return hit_rate
    
    
    def clean_df(self, x):
        
        if x == -99999:
            return ""
        
        else:
            return re.sub(r"\s+", " ", str(x))


    def validate_address(self):
        '''
        Validate addresses
        '''
        addr = self.df.copy()
        addr = addr[(addr['p_inpclnaddrfullflag'] == 0) & (addr['p_inpacct'] != -99999)]
        
        if addr.shape[0] > 0:
            addr = addr[['p_inpacct', 'p_inpaddrline1', 'p_inpaddrline2', 'p_inpaddrcity', 'p_inpaddrstate', 'p_inpaddrzip']].sample(min(10, addr.shape[0]))
            # cleaning address
            addr_clean = addr.applymap(self.clean_df)
            addr_clean['Provided Address'] = addr_clean['p_inpaddrline1'] + ' ' + addr_clean['p_inpaddrline2'] + ' ' + addr_clean['p_inpaddrcity'] + ' ' + addr_clean['p_inpaddrstate'] + ' ' + addr_clean['p_inpaddrzip']
            addr_clean = addr_clean.rename(columns={'p_inpacct': 'Account'})
            addr_clean = addr_clean[['Account', 'Provided Address']]
            return addr_clean
        
        else:
            return "No address validation issues detected in dataset."
        
    
    def state_distribution(self):
        '''
        Get state distribution
        '''
        ds = self.df.copy()
        state_count = ds.loc[~ds['p_inpclnaddrstate'].isin([-99999, -99998]), 'p_inpclnaddrstate'].value_counts().rename_axis('State').reset_index(name='Count')        
        state_count = state_count.rename(columns={'p_inpclnaddrstate': 'State'})

        # plot out the distribution in map
        fig = px.choropleth(state_count, locations='State', locationmode='USA-states', color='Count', scope='usa')
        fig.show()

    
    def top_states(self, num = 10):
        '''
        Get top *num* states with most PII info, defaults to 10
        '''
        ds = self.df.copy()
        state_count = ds.loc[~ds['p_inpclnaddrstate'].isin([-99999, -99998]), 'p_inpclnaddrstate'].value_counts().rename_axis('State').reset_index(name='Count')        
        state_count = state_count.rename(columns={'p_inpclnaddrstate': 'State'})
        state_count = state_count.sort_values(by='Count', ascending=False).head(num)
        # state_count = state_count.reset_index()
        # state_count.columns = ['State', 'Count']

        return state_count
    
    
    def validate_name(self):
        '''
        Validate names
        '''
        name = self.df.copy()
        name = name[(name['p_inpclnnamefirstflag'] == 0) & (name['p_inpclnnamelastflag'] == 0) & (name['p_inpacct'] != -99999)]

        if name.shape[0] > 0:
            # cleaning names
            name = name[['p_inpacct', 'p_inpnamefirst', 'p_inpnamelast']].sample(min(10, name.shape[0]))
            name_clean = name.applymap(self.clean_df)
            name_clean['Provided First Name'] = name_clean['p_inpnamefirst']
            name_clean['Provided Last Name'] = name_clean['p_inpnamelast']
            name_clean = name_clean.rename(columns={'p_inpacct': 'Account'})
            name_clean = name_clean[['Account', 'Provided First Name', 'Provided Last Name']]
            return name_clean
        
        else:
            return "No name validation issues detected in dataset."
        
    
    def validate_DOB(self):
        '''
        Validate dates of birth
        Returns:
            DataFrame: A DataFrame containing the account number and provided date of birth for each record with date of birth validation issues.
            str: A message indicating that no date of birth validation issues were detected in the dataset.
        '''
        dob = self.df.copy()
        dob = dob[(dob['p_inpclndobflag'] == 0) & (dob['p_inpacct'] != -99999)]
        
        if dob.shape[0] > 0:
            dob = dob[['p_inpacct', 'p_inpdob']].sample(min(10, dob.shape[0]))
            return dob
        
        else:
            return "No name validation issues detected in dataset."


    def plot_age_distribution(self):
        age = self.df.copy()
        age = age[age['pi_inpdobage'] > 0]
        
        fig = px.histogram(age, x='pi_inpdobage', nbins=20)
        fig.update_layout(xaxis_title='Age', yaxis_title='Count')
        fig.show()


    def get_age_distribution(self):
        '''
        Get age distribution based on DOB
        '''
        age = self.df.copy()
        age = pd.DataFrame(age['pi_inpdobage'])
        # subgroup age by pi_inpdobage into -99999 -> unavailable, <18 -> <18, 18 -99, 100+
        age['age_group'] = pd.cut(age['pi_inpdobage'], bins=[-100000, 0, 18, 99, 10000], labels=['Unavailable', '<18', '18-99', '100+'])
        # group by age_group and count and get precentage and return the dataframe
        age_count = age.groupby('age_group').count()
        age_count = age_count.rename(columns={'pi_inpdobage': 'Count'})
        age_count['Percentage'] = age_count['Count'] / age_count['Count'].sum() * 100
        age_count = age_count.reset_index()

        return age_count


    def validate_phone(self):
        '''
        Validate phone numbers
        '''
        phone = self.df.copy()
        phone = phone[(phone['p_inpclnphonehomeflag'] == 0) & (phone['p_inpacct'] != -99999)]
        
        if phone.shape[0] > 0:
            phone = phone[['p_inpacct', 'p_inpphonehome']].sample(min(10, phone.shape[0]))
            return phone

        else:
            return "No phone validation issues detected in dataset."
        

    def validate_ssn(self):
        '''
        Validate SSN
        '''
        ssn = self.df.copy()
        ssn = ssn[(ssn['p_inpclnssnflag'] == 0) & (ssn['p_inpacct'] != -99999)]
        
        if ssn.shape[0] > 0:
            ssn = ssn[['p_inpacct', 'p_inpssn']].sample(min(10, ssn.shape[0]))
            ssn_clean = ssn.applymap(self.clean_df)
            ssn_clean['Provided SSN'] = ssn_clean['p_inpssn']
            ssn_clean = ssn_clean.rename(columns={'p_inpacct': 'Account'})
            ssn_clean = ssn_clean[['Account', 'Provided SSN']]
            return ssn_clean
        
        else:
            return "No SSN validation issues detected in dataset."
        

    def ssn_is_itin_flag(self):
        '''
        A flag indicating whether input SSN is likely an ITIN.  The following are characteristics of a ITIN when the first digit is 9:
        '''
        flag = self.df.copy()
        result = (
            flag.groupby('p_inpvalssnisitinflag')
            .size()
            .reset_index(name='count')
            .assign(percent_of_records=lambda df: df['count'] / len(flag))
            .rename(columns={'count': '# of Records'})
            [['p_inpvalssnisitinflag', '# of Records', 'percent_of_records']]
        )

        return result
    

    def ssn_is_itin_sample(self):
        '''
        A flag indicating whether input SSN is likely an ITIN.  The following are characteristics of a ITIN when the fourth and fifth digits have '50' - '65', '70' - '88', '90' - '92', or '94' - '99' 
        ''' 
        try:
            flag = self.df.copy()
            result = (
                flag[flag['p_inpvalssnisitinflag'] == 1][['p_inpacct', 'p_inpssn']]
                .rename(columns={'p_inpacct': 'Account', 'p_inpssn': 'Provided SSN'})
                .sample(min(10, flag.shape[0]))
            )

            return result
        
        except:
            print("Likely ITIN validation issues detected in dataset.")
    

    def invalid_ssn_flag(self):
        '''
        A flag indicating whether input SSN is invalid according to Social Security Administration standards. SSA will not issue a SSN with at least one of the following patterns:

            1) Starts with '9', '000', or '666'

            2) Has '00' at fourth to fifth positions

            3) Has '000' at sixth to ninth positions
        '''
        flag = self.df.copy()
        result = (
            flag.groupby('p_inpvalssnnonssaflag')
            .size()
            .reset_index(name='count')
            .assign(percent_of_records=lambda df: df['count'] / len(flag))
            .rename(columns={'count': '# of Records'})
            [['p_inpvalssnnonssaflag', '# of Records', 'percent_of_records']]
        )

        return result
    
    
    def invalid_ssn_sample(self):
        '''
        A flag indicating whether input SSN is invalid according to Social Security Administration standards. SSA will not issue a SSN with at least one of the following patterns:
        '''
        try:
            flag = self.df.copy()
            result = (
                flag[flag['p_inpvalssnnonssaflag'] == 1][['p_inpacct', 'p_inpssn']]
                .rename(columns={'p_inpacct': 'Account', 'p_inpssn': 'Provided SSN'})
                .sample(min(10, flag.shape[0]))
            )

            return result
        
        except:
            print("Likely Invalid SSN issues detected in dataset based on SSA standards.")
            
    
    def duplicate_PII(self):
        '''
        A flag indicating whether input PII is a duplicate of another PII in the dataset.
        '''
        dupes = self.identify_duplicates()
        dupe_cols = dupes.filter(regex='^Duplicated')
        dupe_counts = dupe_cols.sum()
        dupe_percents = (dupe_cols.mean() * 100).apply(lambda x: f'{x:.2f}%')
        duplicate_sum = pd.DataFrame({'Count': dupe_counts, 'Hit_Rate': dupe_percents})
        duplicate_sum.index.name = 'PII_field'
        duplicate_sum.reset_index(inplace=True)
        return duplicate_sum
    
    
    def duplicate_address(self):
        '''
        A flag indicating whether input address is a duplicate of another address in the dataset.
        '''
        dupes = self.identify_duplicates()
        address_dupes = pd.DataFrame(dupes[dupes['Duplicated Address']]['Address'].drop_duplicates().sample(min(5, dupes.shape[0])))

        data = self.df.copy()
        data = data[data['p_inpclnaddrfull'].isin(address_dupes['Address'])]
        data = data[['p_inpacct', 'p_inpclnaddrfull']].rename(columns={'p_inpacct': 'Account', 'p_inpclnaddrfull': 'Address'})
        data = data.sort_values(by=['Address'])

        return data
    
    def duplicate_address_name(self):
        '''
        A flag indicating whether input address and name is a duplicate of another address and name in the dataset.
        '''
        dupes = self.identify_duplicates()
        address_name_dupes = pd.DataFrame(dupes[dupes['Duplicated Address + Name']]['Address'].drop_duplicates().sample(min(5, dupes.shape[0])))

        data = self.df.copy()
        data = data[data['p_inpclnaddrfull'].isin(address_name_dupes['Address'])]
        data = data[['p_inpacct', 'p_inpclnaddrfull', 'p_inpclnnamefirst', 'p_inpclnnamelast']].rename(columns={'p_inpacct': 'Account', 'p_inpclnaddrfull': 'Address', 'p_inpclnnamefirst': 'First Name', 'p_inpclnnamelast': 'Last Name'})
        data = data.sort_values(by=['Address'])

        return data


    def duplicate_phone(self):
        '''
        A flag indicating whether input phone is a duplicate of another phone in the dataset.
        '''
        dupes = self.identify_duplicates()
        phone_dupes = pd.DataFrame(dupes[dupes['Duplicated Phone']]['Phone'].drop_duplicates().sample(min(5, dupes.shape[0])))

        data = self.df.copy()
        data = data[data['p_inpclnphonehome'].isin(phone_dupes['Phone'])]
        data = data[['p_inpacct', 'p_inpclnphonehome']].rename(columns={'p_inpacct': 'Account', 'p_inpclnphonehome': 'Phone'})
        data = data.sort_values(by=['Phone'])

        return data
    

    def duplicate_phone_name(self):
        '''
        A flag indicating whether input phone and name is a duplicate of another phone and name in the dataset.
        '''
        dupes = self.identify_duplicates()
        phone_name_dupes = pd.DataFrame(dupes[dupes['Duplicated Phone + Name']]['Phone'].drop_duplicates().sample(min(5, dupes.shape[0])))

        data = self.df.copy()
        data = data[data['p_inpclnphonehome'].isin(phone_name_dupes['Phone'])]
        data = data[['p_inpacct', 'p_inpclnphonehome', 'p_inpclnnamefirst', 'p_inpclnnamelast']].rename(columns={'p_inpacct': 'Account', 'p_inpclnphonehome': 'Phone', 'p_inpclnnamefirst': 'First Name', 'p_inpclnnamelast': 'Last Name'})
        data = data.sort_values(by=['Phone'])

        return data
    

    def duplicate_SSN(self):
        '''
        A flag indicating whether input SSN is a duplicate of another SSN in the dataset.
        '''
        dupes = self.identify_duplicates()
        ssn_dupes = pd.DataFrame(dupes[dupes['Duplicated SSN']]['SSN'].drop_duplicates().sample(min(5, dupes.shape[0])))
        
        data = self.df.copy()
        data = data[data['p_inpclnssn'].isin(ssn_dupes['SSN'])]
        data = data[['p_inpacct', 'p_inpclnssn']].rename(columns={'p_inpacct': 'Account', 'p_inpclnssn': 'SSN'})
        data = data.sort_values(by=['SSN'])

        return data
    

    def duplicate_SSN_name(self):
        '''
        A flag indicating whether input SSN and name is a duplicate of another SSN and name in the dataset.
        '''
        dupes = self.identify_duplicates()
        ssn_name_dupes = pd.DataFrame(dupes[dupes['Duplicated SSN + Name']]['SSN'].drop_duplicates().sample(min(5, dupes.shape[0])))

        data = self.df.copy()
        data = data[data['p_inpclnssn'].isin(ssn_name_dupes['SSN'])]
        data = data[['p_inpacct', 'p_inpclnssn', 'p_inpclnnamefirst', 'p_inpclnnamelast']].rename(columns={'p_inpacct': 'Account', 'p_inpclnssn': 'SSN', 'p_inpclnnamefirst': 'First Name', 'p_inpclnnamelast': 'Last Name'})
        data = data.sort_values(by=['SSN'])

        return data

