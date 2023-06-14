#| label: input data
import pandas as pd
import numpy as np  
from ydata_profiling import ProfileReport
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
import streamlit.components.v1 as components
import eda_tools.missing_analysis as ma
import eda_tools.EDA_tool as eda

df = pd.read_csv('test.csv', index_col = 0)
st.dataframe(df)

# Profile Report
profile = ProfileReport(df, explorative=False, config_file="profile_config/yprofile_config_default.yaml")
st_profile_report(profile)
#components.html(profile.to_notebook_iframe())
