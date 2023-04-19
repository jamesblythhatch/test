#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time

start_time = time.time()

import pandas as pd
import numpy as np
import panel as pn
import math as math
import xarray as xr
import geoviews as gv
from datetime import datetime as dt
from datetime import date, timedelta
pn.extension('tabulator')

import hvplot.pandas
import hvplot.xarray
import hvplot as hvplot

import cartopy.crs as ccrs

import holoviews as hv
from holoviews import opts, dim

import geopandas as gpd

import requests

import os
import json
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import io
import pickle

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Library import time elapsed: {elapsed_time:.2f} seconds")


# In[2]:


def fetch_data():
    start_time = time.time()
    # Set the scopes
    SCOPES = ["https://www.googleapis.com/auth/drive"]

    # Load credentials from the client_secrets dictionary
    client_secrets = json.loads(os.environ['GOOGLE_API_CREDENTIALS'])

    # Authenticate using the client_secrets
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_config(client_secrets, SCOPES)
            auth_url, _ = flow.authorization_url(prompt="consent", access_type='offline')
            print(f"Please visit this URL to authorize the application: {auth_url}")
            authorization_response = input("Enter the full callback URL: ")
            flow.fetch_token(authorization_response=authorization_response)
            creds = flow.credentials
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    # Create the Google Drive API client
    service = build("drive", "v3", credentials=creds)

    def read_csv_from_drive(service, file_id):
        try:
            request = service.files().get_media(fileId=file_id)
            file_data = io.BytesIO()
            downloader = MediaIoBaseDownload(file_data, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print("Download %d%%." % int(status.progress() * 100))

            file_data.seek(0)
            df = pd.read_csv(file_data, skiprows=4, header=None)
            return df
        except HttpError as error:
            print(f"An error occurred: {error}")
            return None

    # Replace these with your own file IDs
    file_id_1 = "1Yi83mmUHnrKZ3BYv2BKTpPUY5s5T6zCx"
    file_id_2 = "1_IWfzZD4ONsmnJnAi3Om1Q0-usIhRWeU"

    datalogger11_dataframe = read_csv_from_drive(service, file_id_1)
    datalogger55_dataframe = read_csv_from_drive(service, file_id_2)
    
    #df_water = pd.read_csv('./files/H368314_water_level.csv', skiprows=9)
    start_date = '2023-04-08'
    current_date = str(date.today())
    water_start_time = '00:00:00'
    water_current_time = dt.now() + timedelta(hours=4)
    water_current_time_str = str(water_current_time.strftime("%H:%M:%S"))

    url = 'https://wateroffice.ec.gc.ca/services/real_time_data/csv/inline?stations[]=01AP005&parameters[]=46&start_date=' + start_date + '%20' + water_start_time + '&end_date=' + current_date + '%20' + water_current_time_str

    df_water = pd.read_csv(url)
    df_water['Date'] = pd.to_datetime(df_water['Date'], format="%Y-%m-%dT%H:%M:%SZ")
    df_water['Date'] = df_water['Date'] - timedelta(hours=5)
    df_water['Value/Valeur'] = df_water['Value/Valeur'] - 0.075

    #read in the latest survey table of data
    df_raw = pd.read_csv('./files/H368314_all_monitoring_data.csv')

    #read in NAD locations
    df_nad = pd.read_csv('./files/H368314_survey_point_NAD83.csv')
    df_piezo_nad = pd.read_csv('./files/H368314_piezo_points.csv')

    #read in piezos bulk
    column_names = ['Date', 'VW1_B', 'VW1_T', 'VW2_B', 'VW2_T']
    column_names_single = ['Date', 'VW1_B', 'VW1_T']
    df_bh202 = datalogger55_dataframe[[0,14,15,16,17]]
    df_bh204 = datalogger55_dataframe[[0,4,5,6,7]]
    df_bh03 = datalogger55_dataframe[[0,24,25,26,27]]
    df_bh05 = datalogger55_dataframe[[0,34,35,36,37]]
    df_bh02 = datalogger11_dataframe[[0,6,7]]
    df_bh10 = datalogger11_dataframe[[0,4,5]]
    df_bh06 = datalogger11_dataframe[[0,8,9]]

    df_bh202.columns = column_names
    df_bh204.columns = column_names
    df_bh03.columns = column_names
    df_bh05.columns = column_names
    df_bh02.columns = column_names_single
    df_bh10.columns = column_names_single
    df_bh06.columns = column_names_single

    df_bh202[['VW1_B', 'VW1_T', 'VW2_B', 'VW2_T']] = df_bh202[['VW1_B', 'VW1_T', 'VW2_B', 'VW2_T']].astype(float)
    df_bh204[['VW1_B', 'VW1_T', 'VW2_B', 'VW2_T']] = df_bh204[['VW1_B', 'VW1_T', 'VW2_B', 'VW2_T']].astype(float)
    df_bh03[['VW1_B', 'VW1_T', 'VW2_B', 'VW2_T']] = df_bh03[['VW1_B', 'VW1_T', 'VW2_B', 'VW2_T']].astype(float)
    df_bh05[['VW1_B', 'VW1_T', 'VW2_B', 'VW2_T']] = df_bh05[['VW1_B', 'VW1_T', 'VW2_B', 'VW2_T']].astype(float)
    df_bh02[['VW1_B', 'VW1_T']] = df_bh02[['VW1_B', 'VW1_T']].astype(float)
    df_bh10[['VW1_B', 'VW1_T']] = df_bh10[['VW1_B', 'VW1_T']].astype(float)
    df_bh06[['VW1_B', 'VW1_T']] = df_bh06[['VW1_B', 'VW1_T']].astype(float)

    #min date to filter piezos
    min_date = str(df_water['Date'].min())

    #filter the piezo dataframes
    df_bh202 = df_bh202[df_bh202['Date'] > min_date]
    df_bh204 = df_bh204[df_bh204['Date'] > min_date]
    df_bh03 = df_bh03[df_bh03['Date'] > min_date]
    df_bh05 = df_bh05[df_bh05['Date'] > min_date]
    df_bh02 = df_bh02[df_bh02['Date'] > min_date]
    df_bh10 = df_bh10[df_bh10['Date'] > min_date]
    df_bh06 = df_bh06[df_bh06['Date'] > min_date]

    dataframes = [df_bh202, df_bh204, df_bh02, df_bh03, df_bh05, df_bh06, df_bh10]
    dataframes_twopiez = [df_bh202, df_bh204, df_bh03, df_bh05]

    for df in dataframes:
        df.loc[df['VW1_B'] == 0, 'VW1_B'] = np.nan
        df.loc[df['VW1_B'] == 0, 'VW1_T'] = np.nan

    for df in dataframes_twopiez:
        df.loc[df['VW2_B'] == 0, 'VW2_B'] = np.nan
        df.loc[df['VW2_B'] == 0, 'VW2_T'] = np.nan

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Reading data time elapsed: {elapsed_time:.2f} seconds")
    return df_water, df_raw, df_nad, df_piezo_nad, df_bh202, df_bh204, df_bh02, df_bh03, df_bh05, df_bh06, df_bh10


# In[3]:


def process_data(df_water, df_raw, df_nad, df_piezo_nad, df_bh202, df_bh204, df_bh02, df_bh03, df_bh05, df_bh06, df_bh10):

    start_time = time.time()

    #calculate difference between each row in dataframe
    df_diff = df_raw.loc[:, df_raw.columns != 'Date'].diff() #date column is not included to avoid math error

    #put the date column back
    extracted_column = df_raw["Date"]
    df_diff.insert(0,"Date",extracted_column)

    #convert to metric mm, same as before date removed
    df_diff_mm = df_diff.select_dtypes(exclude=['object', 'datetime']) * 0.3048 * 1000
    extracted_column = df_raw["Date"]
    df_diff_mm.insert(0,"Date",extracted_column)

    #create the total translational movement column
    for column in df_diff_mm:
        if column[-2:] == '_N': #equivalent of right in VBA
            east_col = column[:-2] + '_E'
            new_col = np.sqrt(df_diff_mm[column]**2 + df_diff_mm[east_col]**2)
            new_col_name = column[:-2] + '_T' #string concat is with + in Python
            new_col_index = df_diff_mm.columns.get_loc(column) + 3 
            df_diff_mm.insert(new_col_index,new_col_name,new_col) #insert calculated column into table

    #compute the directional angle of the movement (polar)
    for column in df_diff_mm:
        if column[-2:] == '_N':
            east_col = column[:-2] + '_E'
            new_col = np.degrees(np.arctan2(df_diff_mm[east_col],df_diff_mm[column]))
            new_col = new_col + 20.8668131869825
            new_col = np.where(new_col < 0, new_col + 360, new_col)
            new_col_name = column[:-2] + '_angle'
            new_col_index = df_diff_mm.columns.get_loc(column) + 4
            df_diff_mm.insert(new_col_index,new_col_name,new_col)

    #remove NaN values and display the edited dataframe
    df_diff_mm = df_diff_mm.fillna(0)
    
    #convert to metric mm, same as before date removed
    df_total_mm = df_raw.select_dtypes(exclude=['object', 'datetime']) * 0.3048 * 1000
    length = len(df_total_mm.index)

    #first calculate the total movements for the simple columns with data in the first row
    cols = []
    for column in df_total_mm:
        if math.isnan(df_total_mm[column].iloc[0]) == False:
            cols.append(column)
    for column in cols:
        df_total_mm[column] = df_total_mm[column] - df_total_mm[column].iloc[0]

    #then do the same for the columns without data in the first row
    break_i = False
    for i in reversed(range(1, length-1)):
        cols = []
        for column in df_total_mm:
            if math.isnan(df_total_mm[column].iloc[i]) == True and math.isnan(df_total_mm[column].iloc[i+1]) == False:
                cols.append(column)
        for column in cols:
            df_total_mm[column] = df_total_mm[column] - df_total_mm[column].iloc[i+1]

    #create the total translational movement column
    for column in df_total_mm:
        if column[-2:] == '_N': #equivalent of right in VBA
            east_col = column[:-2] + '_E'
            new_col = np.sqrt(df_total_mm[column]**2 + df_total_mm[east_col]**2)
            new_col_name = column[:-2] + '_T' #string concat is with + in Python
            new_col_index = df_total_mm.columns.get_loc(column) + 3 
            df_total_mm.insert(new_col_index,new_col_name,new_col) #insert calculated column into table

    #compute the directional angle of the movement (polar)
    for column in df_total_mm:
        if column[-2:] == '_N':
            east_col = column[:-2] + '_E'
            new_col = np.degrees(np.arctan2(df_total_mm[east_col],df_total_mm[column]))
            new_col = new_col + 20.8668131869825
            new_col = np.where(new_col < 0, new_col + 360, new_col)
            new_col_name = column[:-2] + '_angle'
            new_col_index = df_total_mm.columns.get_loc(column) + 4
            df_total_mm.insert(new_col_index,new_col_name,new_col)

    #put the date column back and fill the NaN values with zero
    extracted_column = df_raw["Date"]
    df_total_mm.insert(0,"Date",extracted_column)
    df_total_mm = df_total_mm.fillna(0)
    
    #dictionary of conversion factors for piezo calc
    piezo_factors = {
        'BH-02': {
            'VW1': {
                'L0': 8942.0,
                'T0': 2.9,
                'Tk': -0.054673,
                'CF': 0.11758,
                'A': 0.00000026979,
                'B': -0.12160
            },
        },
        'BH-10': {
            'VW1': {
                'L0': 8986.9,
                'T0': 3.5,
                'Tk': -0.037830,
                'CF': 0.10055,
                'A': -0.000000042938,
                'B': -0.099923
            },
        },
        'BH-03': {
            'VW1': {
                'L0': 9242.5,
                'T0': 11.2,
                'Tk': -0.080886,
                'CF': 0.11412,
                'A': 0.00000019716,
                'B': -0.11716
            },
            'VW2': {
                'L0': 8926.1,
                'T0': 17.3,
                'Tk': -0.050528,
                'CF': 0.12103,
                'A': 0.000000032950,
                'B': -0.12152
            }
        },
        'BH-05': {
            'VW1': {
                'L0': 8906.7,
                'T0': 9.2,
                'Tk': -0.061786,
                'CF': 0.099654,
                'A': 0.00000030675,
                'B': -0.10404
            },
            'VW2': {
                'L0': 8693.9,
                'T0': 10.7,
                'Tk': -0.10242,
                'CF': 0.11117,
                'A': 0.00000014563,
                'B': -0.11325            
            }
        },
        'BH-06': {
            'VW1': {
                'L0': 8551.7,
                'T0': 3.3,
                'Tk': -0.077811,
                'CF': 0.098284,
                'A': 0.0000000021590,
                'B': -0.098213
            }
        },
        'BH-202': {
            'VW1': {
                'L0': 8828.9,
                'T0': 29.7,
                'Tk': 0.15747,
                'CF': 0.092428
            },
            'VW2': {
                'L0': 8571.3,
                'T0': 30.3,
                'Tk': 0.13326,
                'CF': 0.19454
            }
        },
        'BH-204': {
            'VW1': {
                'L0': 8516.7,
                'T0': 23.2,
                'Tk': 0.13094,
                'CF': 0.097717
            },
            'VW2': {
                'L0': 8451.2,
                'T0': 24.1,
                'Tk': -0.15697,
                'CF': 0.19114
            }
        }
    }       

    #run all the piezo calculations, new piezos use A(L²) + B(L) + C -Tk(T0 - T), old piezos use CF(L0 - L) - Tk(T0 - T)
    #C = -[A(L0²) + B(L0)]
    bh202_dict = piezo_factors['BH-202']['VW1']
    df_bh202['VW1'] = (bh202_dict['CF']*(bh202_dict['L0'] - df_bh202['VW1_B'])) - (bh202_dict['Tk']*(bh202_dict['T0'] - df_bh202['VW1_T']))

    bh202_dict = piezo_factors['BH-202']['VW2']
    df_bh202['VW2'] = (bh202_dict['CF']*(bh202_dict['L0'] - df_bh202['VW2_B'])) - (bh202_dict['Tk']*(bh202_dict['T0'] - df_bh202['VW2_T']))

    bh204_dict = piezo_factors['BH-204']['VW1']
    df_bh204['VW1'] = (bh204_dict['CF']*(bh204_dict['L0'] - df_bh204['VW1_B'])) - (bh204_dict['Tk']*(bh204_dict['T0'] - df_bh204['VW1_T']))

    bh204_dict = piezo_factors['BH-204']['VW2']
    df_bh204['VW2'] = (bh204_dict['CF']*(bh204_dict['L0'] - df_bh204['VW2_B'])) - (bh204_dict['Tk']*(bh204_dict['T0'] - df_bh204['VW2_T']))

    bh03_dict = piezo_factors['BH-03']['VW1']
    df_bh03['VW1'] = (bh03_dict['A']*(df_bh03['VW1_B'] ** 2)) + (bh03_dict['B'] * df_bh03['VW1_B']) - ((bh03_dict['A']*(bh03_dict['L0'] ** 2)) + (bh03_dict['B']*(bh03_dict['L0']))) - (bh03_dict['Tk']*(bh03_dict['T0'] - df_bh03['VW1_T']))

    bh03_dict = piezo_factors['BH-03']['VW2']
    df_bh03['VW2'] = (bh03_dict['A']*(df_bh03['VW2_B'] ** 2)) + (bh03_dict['B'] * df_bh03['VW2_B']) - ((bh03_dict['A']*(bh03_dict['L0'] ** 2)) + (bh03_dict['B']*(bh03_dict['L0']))) - (bh03_dict['Tk']*(bh03_dict['T0'] - df_bh03['VW2_T']))

    bh05_dict = piezo_factors['BH-05']['VW1']
    df_bh05['VW1'] = (bh05_dict['A']*(df_bh05['VW1_B'] ** 2)) + (bh05_dict['B'] * df_bh05['VW1_B']) - ((bh05_dict['A']*(bh05_dict['L0'] ** 2)) + (bh05_dict['B']*(bh05_dict['L0']))) - (bh05_dict['Tk']*(bh05_dict['T0'] - df_bh05['VW1_T']))

    bh05_dict = piezo_factors['BH-05']['VW2']
    df_bh05['VW2'] = (bh05_dict['A']*(df_bh05['VW2_B'] ** 2)) + (bh05_dict['B'] * df_bh05['VW2_B']) - ((bh05_dict['A']*(bh05_dict['L0'] ** 2)) + (bh05_dict['B']*(bh05_dict['L0']))) - (bh05_dict['Tk']*(bh05_dict['T0'] - df_bh05['VW2_T']))

    bh02_dict = piezo_factors['BH-02']['VW1']
    df_bh02['VW1'] = (bh02_dict['A']*(df_bh02['VW1_B'] ** 2)) + (bh02_dict['B'] * df_bh02['VW1_B']) - ((bh02_dict['A']*(bh02_dict['L0'] ** 2)) + (bh02_dict['B']*(bh02_dict['L0']))) - (bh02_dict['Tk']*(bh02_dict['T0'] - df_bh02['VW1_T']))

    bh10_dict = piezo_factors['BH-10']['VW1']
    df_bh10['VW1'] = (bh10_dict['A']*(df_bh10['VW1_B'] ** 2)) + (bh10_dict['B'] * df_bh10['VW1_B']) - ((bh10_dict['A']*(bh10_dict['L0'] ** 2)) + (bh10_dict['B']*(bh10_dict['L0']))) - (bh10_dict['Tk']*(bh10_dict['T0'] - df_bh10['VW1_T']))

    bh06_dict = piezo_factors['BH-06']['VW1']
    df_bh06['VW1'] = (bh06_dict['A']*(df_bh06['VW1_B'] ** 2)) + (bh06_dict['B'] * df_bh06['VW1_B']) - ((bh06_dict['A']*(bh06_dict['L0'] ** 2)) + (bh06_dict['B']*(bh06_dict['L0']))) - (bh06_dict['Tk']*(bh06_dict['T0'] - df_bh06['VW1_T']))

    #Reorganise dataframes so they are more amenable to the widgets
    df_diff_mm = df_diff_mm.melt(id_vars=["Date"], var_name="Point_atr", value_name="Value")
    df_diff_mm[['Point','Measurement']] = df_diff_mm.Point_atr.str.split("_",expand=True)
    df_diff_mm = df_diff_mm.drop('Point_atr', axis=1)
    df_total_mm = df_total_mm.melt(id_vars=["Date"], var_name="Point_atr", value_name="Value")
    df_total_mm[['Point','Measurement']] = df_total_mm.Point_atr.str.split("_",expand=True)
    df_total_mm = df_total_mm.drop('Point_atr', axis=1)
    df_water = df_water.rename(columns={'Value/Valeur': 'River'})
    df_bh202 = df_bh202.drop(['VW1_B', 'VW1_T', 'VW2_B', 'VW2_T'], axis=1)
    df_bh204 = df_bh204.drop(['VW1_B', 'VW1_T', 'VW2_B', 'VW2_T'], axis=1)
    df_bh02 = df_bh02.drop(['VW1_B', 'VW1_T'], axis=1)
    df_bh06 = df_bh06.drop(['VW1_B', 'VW1_T'], axis=1)
    df_bh10 = df_bh10.drop(['VW1_B', 'VW1_T'], axis=1)
    df_bh03 = df_bh03.drop(['VW1_B', 'VW1_T', 'VW2_B', 'VW2_T'], axis=1)
    df_bh05 = df_bh05.drop(['VW1_B', 'VW1_T', 'VW2_B', 'VW2_T'], axis=1)
    df_water = df_water.drop([' ID', 'Parameter/Paramètre', 'Qualifier/Qualificatif', 'Symbol/Symbole', 'Approval/Approbation'], axis=1)
    df_water['Date'] = pd.to_datetime(df_water['Date'])
    df_bh202['Date'] = pd.to_datetime(df_bh202['Date'])
    df_bh204['Date'] = pd.to_datetime(df_bh204['Date'])
    df_bh02['Date'] = pd.to_datetime(df_bh02['Date'])
    df_bh03['Date'] = pd.to_datetime(df_bh03['Date'])
    df_bh05['Date'] = pd.to_datetime(df_bh05['Date'])
    df_bh06['Date'] = pd.to_datetime(df_bh06['Date'])
    df_bh10['Date'] = pd.to_datetime(df_bh10['Date'])

    df_water['River'] = df_water['River'].astype(float)
    df_water['River'] = df_water['River'] - (1.19 * 0.3048) #convert to plant datum
    
    #sort the dataframes
    df_bh202 = df_bh202.sort_values(by='Date')
    df_bh204 = df_bh204.sort_values(by='Date')
    df_bh02 = df_bh02.sort_values(by='Date')
    df_bh03 = df_bh03.sort_values(by='Date')
    df_bh05 = df_bh05.sort_values(by='Date')
    df_bh06 = df_bh06.sort_values(by='Date')
    df_bh10 = df_bh10.sort_values(by='Date')

    #merge the river data
    df_bh202 = pd.merge_asof(df_bh202, df_water, on='Date', direction='nearest')
    df_bh204 = pd.merge_asof(df_bh204, df_water, on='Date', direction='nearest')
    df_bh02 = pd.merge_asof(df_bh02, df_water, on='Date', direction='nearest')
    df_bh03 = pd.merge_asof(df_bh03, df_water, on='Date', direction='nearest')
    df_bh05 = pd.merge_asof(df_bh05, df_water, on='Date', direction='nearest')
    df_bh06 = pd.merge_asof(df_bh06, df_water, on='Date', direction='nearest')
    df_bh10 = pd.merge_asof(df_bh10, df_water, on='Date', direction='nearest')
    
    df_bh202['VW1_excesspp'] = df_bh202['VW1'] - 9.81*(df_bh202['River'] - (3.15 - 13.3))
    df_bh202['VW2_excesspp'] = df_bh202['VW2'] - 9.81*(df_bh202['River'] - (3.15 - 23.9))
    df_bh204['VW1_excesspp'] = df_bh204['VW1'] - 9.81*(df_bh204['River'] - (1.61 - 10.7))
    df_bh204['VW2_excesspp'] = df_bh204['VW2'] - 9.81*(df_bh204['River'] - (1.61 - 16.5))
    df_bh02['VW1_excesspp'] = df_bh02['VW1'] - 9.81*(df_bh02['River'] - (1.30 - 6.43))
    df_bh03['VW1_excesspp'] = df_bh03['VW1'] - 9.81*(df_bh03['River'] - (1.55 - 5.33))
    df_bh03['VW2_excesspp'] = df_bh03['VW2'] - 9.81*(df_bh03['River'] - (1.55 - 10.06))
    df_bh05['VW1_excesspp'] = df_bh05['VW1'] - 9.81*(df_bh05['River'] - (1.26 - 6.63))
    df_bh05['VW2_excesspp'] = df_bh05['VW2'] - 9.81*(df_bh05['River'] - (1.26 - 9.85))
    df_bh06['VW1_excesspp'] = df_bh06['VW1'] - 9.81*(df_bh06['River'] - (2.95 - 9.37))
    df_bh10['VW1_excesspp'] = df_bh10['VW1'] - 9.81*(df_bh10['River'] - (1.27 - 9.20))

    df_bh202['River'] = df_bh202['River'] * 9.81
    df_bh204['River'] = df_bh204['River'] * 9.81
    df_bh02['River'] = df_bh02['River'] * 9.81
    df_bh03['River'] = df_bh03['River'] * 9.81
    df_bh05['River'] = df_bh05['River'] * 9.81
    df_bh06['River'] = df_bh06['River'] * 9.81
    df_bh10['River'] = df_bh10['River'] * 9.81
    
    df_bh202 = df_bh202.melt(id_vars=["Date"], var_name="VWid", value_name="Pressure")
    df_bh204 = df_bh204.melt(id_vars=["Date"], var_name="VWid", value_name="Pressure")
    df_bh02 = df_bh02.melt(id_vars=["Date"], var_name="VWid", value_name="Pressure")
    df_bh03 = df_bh03.melt(id_vars=["Date"], var_name="VWid", value_name="Pressure")
    df_bh05 = df_bh05.melt(id_vars=["Date"], var_name="VWid", value_name="Pressure")
    df_bh06 = df_bh06.melt(id_vars=["Date"], var_name="VWid", value_name="Pressure")
    df_bh10 = df_bh10.melt(id_vars=["Date"], var_name="VWid", value_name="Pressure")
    #df_water = df_water.melt(id_vars=["Date"], var_name="VWid", value_name="Head")
    
    #convert piezo data to head
    df_bh202["Pressure"] = df_bh202["Pressure"]/9.81
    df_bh204["Pressure"] = df_bh204["Pressure"]/9.81
    df_bh02["Pressure"] = df_bh02["Pressure"]/9.81
    df_bh03["Pressure"] = df_bh03["Pressure"]/9.81
    df_bh05["Pressure"] = df_bh05["Pressure"]/9.81
    df_bh06["Pressure"] = df_bh06["Pressure"]/9.81
    df_bh10["Pressure"] = df_bh10["Pressure"]/9.81

    df_bh202 = df_bh202.rename(columns={'Pressure': 'Head'})
    df_bh204 = df_bh204.rename(columns={'Pressure': 'Head'})
    df_bh02 = df_bh02.rename(columns={'Pressure': 'Head'})
    df_bh03 = df_bh03.rename(columns={'Pressure': 'Head'})
    df_bh05 = df_bh05.rename(columns={'Pressure': 'Head'})
    df_bh06 = df_bh06.rename(columns={'Pressure': 'Head'})
    df_bh10 = df_bh10.rename(columns={'Pressure': 'Head'})
    
    #Add the NAD columns¶
    df_diff_mm = pd.merge(df_diff_mm, df_nad)
    
    df_total_mm = pd.merge(df_total_mm, df_nad)
    
    #Use pivot table to put the measurements as column headings
    df_diff_mm = df_diff_mm.pivot_table(index=['Date', 'Point', 'NADN', 'NADE'],
                                        columns='Measurement',
                                        values='Value',
                                        aggfunc='first').reset_index()
    
    df_total_mm = df_total_mm.pivot_table(index=['Date', 'Point', 'NADN', 'NADE'],
                                          columns='Measurement',
                                          values='Value',
                                          aggfunc='first').reset_index()
    #Create geodataframes
    crs=ccrs.epsg(2953)

    #create geodataframe for the points labels
    gdf_nad = gpd.GeoDataFrame(df_nad, geometry=gpd.points_from_xy(df_nad.NADE, df_nad.NADN, crs=crs))
    gdf_nad = gdf_nad.set_crs(epsg=2953)
    gdf_nad = gdf_nad.to_crs(epsg=4326)

    #create geodataframe for the piezos
    gdf_piezo_nad = gpd.GeoDataFrame(df_piezo_nad, geometry=gpd.points_from_xy(df_piezo_nad.NADE, df_piezo_nad.NADN, crs=crs))
    gdf_piezo_nad = gdf_piezo_nad.set_crs(epsg=2953)
    gdf_piezo_nad = gdf_piezo_nad.to_crs(epsg=4326)

    #create geodataframes for the settlements
    gdf_total_mm = gpd.GeoDataFrame(df_total_mm, geometry=gpd.points_from_xy(df_total_mm.NADE, df_total_mm.NADN, crs=crs))
    gdf_total_mm = gdf_total_mm.set_crs(epsg=2953)
    gdf_total_mm = gdf_total_mm.to_crs(epsg=4326)

    gdf_diff_mm = gpd.GeoDataFrame(df_diff_mm, geometry=gpd.points_from_xy(df_diff_mm.NADE, df_diff_mm.NADN, crs=crs))
    gdf_diff_mm = gdf_diff_mm.set_crs(epsg=2953)
    gdf_diff_mm = gdf_diff_mm.to_crs(epsg=4326)
    
    #Create datasets for the map plots
    df_total_mm_multi = df_total_mm.set_index(['NADN', 'NADE', 'Date', 'Point'])
    ds_total_mm = xr.Dataset.from_dataframe(df_total_mm_multi)

    ds_total_mm = ds_total_mm.assign_attrs({'crs': crs})
    
    df_diff_mm_multi = df_diff_mm.set_index(['NADN', 'NADE', 'Date', 'Point'])
    ds_diff_mm = xr.Dataset.from_dataframe(df_diff_mm_multi)

    ds_diff_mm = ds_diff_mm.assign_attrs({'crs': crs})

    #date time the dataframes
    #date time the date column
    df_diff_mm['Date'] = pd.to_datetime(df_diff_mm['Date']).dt.date
    df_total_mm['Date'] = pd.to_datetime(df_diff_mm['Date']).dt.date
    
    #Make the dataframes interactive
    #interactivisation
    idf_diff_mm = df_diff_mm.interactive()
    idf_total_mm = df_total_mm.interactive()
    idf_water = df_water.interactive()
    idf_bh202 = df_bh202.interactive()
    idf_bh204 = df_bh204.interactive()
    idf_bh02 = df_bh02.interactive()
    idf_bh03 = df_bh03.interactive()
    idf_bh05 = df_bh05.interactive()
    idf_bh06 = df_bh06.interactive()
    idf_bh10 = df_bh10.interactive()
    #igdf_diff_mm = gdf_diff_mm.interactive()
    #igdf_total_mm = gdf_total_mm.interactive()

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Processing data time elapsed: {elapsed_time:.2f} seconds")
    
    return df_water, df_raw, df_nad, df_piezo_nad, df_bh202, df_bh204, df_bh02, df_bh03, df_bh05, df_bh06, df_bh10, df_total_mm, idf_diff_mm, idf_total_mm, idf_bh202, idf_bh204, idf_bh02, idf_bh03, idf_bh05, idf_bh06, idf_bh10, gdf_nad, ds_total_mm, ds_diff_mm, gdf_total_mm, gdf_diff_mm, gdf_piezo_nad, crs


# In[4]:


def create_widgets_plots(df_water, df_raw, df_nad, df_piezo_nad, df_bh202, df_bh204, df_bh02, df_bh03, df_bh05, df_bh06, df_bh10, df_total_mm, idf_diff_mm, idf_total_mm, idf_bh202, idf_bh204, idf_bh02, idf_bh03, idf_bh05, idf_bh06, idf_bh10, gdf_nad, ds_total_mm, ds_diff_mm, gdf_total_mm, gdf_diff_mm, gdf_piezo_nad, crs):
    start_time = time.time()

    #get the upper and lower bounds
    earliest_date = df_total_mm['Date'].min(skipna=False)
    latest_date = df_total_mm['Date'].max(skipna=False)
    piezo_earliest_date = df_water['Date'].min(skipna=False)
    piezo_latest_date = dt.now()

    #create a date slider to filter the data
    upper_date_slider = pn.widgets.DateSlider(name='Upper bound of date range', start=earliest_date, end=latest_date, step=86400000, value=latest_date)
    lower_date_slider = pn.widgets.DateSlider(name='Lower bound of date range', start=earliest_date, end=latest_date, step=86400000)
    piezo_date_slider = pn.widgets.DatetimeRangeSlider(name='Date range', start=piezo_earliest_date, end=piezo_latest_date, step=86400000, value=(piezo_earliest_date, piezo_latest_date))
    #piezo_lower_date_slider = pn.widgets.DateTimeSlider(name='Lower bound of date range', start=piezo_earliest_date, end=piezo_latest_date, step=86400000)

    #radio buttons for each column value we want to examine
    point_list = df_total_mm.Point.unique().tolist()
    point_button = pn.widgets.CheckButtonGroup(name='Displayed Points', options=point_list, button_type = 'success', value=point_list)

    #connect the data to the widgets
    diff_pipeline = (
        idf_diff_mm[
            (idf_diff_mm.Date >= lower_date_slider) &
            (idf_diff_mm.Date <= upper_date_slider) &
            (idf_diff_mm.Point.isin(point_button))
        ]
    )

    total_pipeline = (
        idf_total_mm[
            (idf_total_mm.Date >= lower_date_slider) &
            (idf_total_mm.Date <= upper_date_slider) &
            (idf_total_mm.Point.isin(point_button))
        ]
    )

    bh202_pipeline = (
        idf_bh202[
            (idf_bh202.Date <= piezo_date_slider.param.value_end) &
            (idf_bh202.Date >= piezo_date_slider.param.value_start)
        ]
    )

    bh204_pipeline = (
        idf_bh204[
            (idf_bh204.Date <= piezo_date_slider.param.value_end) &
            (idf_bh204.Date >= piezo_date_slider.param.value_start)
        ]
    )

    bh02_pipeline = (
        idf_bh02[
            (idf_bh02.Date <= piezo_date_slider.param.value_end) &
            (idf_bh02.Date >= piezo_date_slider.param.value_start)
        ]
    )

    bh03_pipeline = (
        idf_bh03[
            (idf_bh03.Date <= piezo_date_slider.param.value_end) &
            (idf_bh03.Date >= piezo_date_slider.param.value_start)
        ]
    )

    bh05_pipeline = (
        idf_bh05[
            (idf_bh05.Date <= piezo_date_slider.param.value_end) &
            (idf_bh05.Date >= piezo_date_slider.param.value_start)
        ]
    )

    bh06_pipeline = (
        idf_bh06[
            (idf_bh06.Date <= piezo_date_slider.param.value_end) &
            (idf_bh06.Date >= piezo_date_slider.param.value_start)
        ]
    )

    bh10_pipeline = (
        idf_bh10[
            (idf_bh10.Date <= piezo_date_slider.param.value_end) &
            (idf_bh10.Date >= piezo_date_slider.param.value_start)
        ]
    )

    #Construct the plots


    diff_plot_T = diff_pipeline.hvplot(x = 'Date', by='Point',
                                         y='T',
                                         ylabel='Translational displacement (mm/day)',
                                         line_width=1,
                                         title='Rate of change of transalational displacement vs. date',
                                         legend=False,
                                         fontscale=0.95,
                                         grid=True,
                                       shared_axes=False
                                        )

    total_plot_T = total_pipeline.hvplot(x = 'Date',
                                           by='Point',
                                           y='T',
                                           ylabel='Translational displacement (mm)',
                                           line_width=1, title='Total translational displacement vs. date',
                                           legend=False,
                                           fontscale=0.95,
                                           grid=True,
                                       shared_axes=False
                                          )

    diff_plot_Z = diff_pipeline.hvplot(x = 'Date',
                                         by='Point',
                                         y='Z',
                                         ylabel='Vertical displacement (mm/day)',
                                         line_width=1,
                                         title='Rate of change of vertical displacement vs. date',
                                         legend=False,
                                         fontscale=0.95,
                                         grid=True,
                                       shared_axes=False
                                        )

    total_plot_Z = total_pipeline.hvplot(x = 'Date',
                                           by='Point',
                                           y='Z',
                                           ylabel='Vertical displacement (mm)',
                                           line_width=1,
                                           title='Total vertical displacement vs. date',
                                           legend=False,
                                           fontscale=0.95,
                                           grid=True,
                                       shared_axes=False
                                          )

    infilling_extents = gpd.read_file('./files/current_infilling.shp')
    infilling_extents_plot = infilling_extents.hvplot(geo=True, tiles='ESRI')

    #labels
    labels_T = gdf_nad.assign(x=lambda gdf: gdf.geometry.x,
                              y=lambda gdf: gdf.geometry.y).hvplot.labels(text="Point",
                                                                          x='x',
                                                                          y='y',
                                                                          geo=True).opts(opts.Labels(text_font_size='9pt',
                                                                                                     xoffset=5,
                                                                                                     yoffset=5))
    labels_Z = gdf_nad.assign(x=lambda gdf: gdf.geometry.x,
                              y=lambda gdf: gdf.geometry.y).hvplot.labels(text="Point",
                                                                          x='x',
                                                                          y='y',
                                                                          geo=True,
                                                                          hover=False).opts(opts.Labels(text_font_size='9pt',
                                                                                                        xoffset=5,
                                                                                                        yoffset=5))

    vectors_total_T = ds_total_mm.hvplot.vectorfield(x='NADE',
                                                     y='NADN',
                                                     angle='angle',
                                                     mag='T',
                                                     by='Point',
                                                     groupby='Date',
                                                     geo=True,
                                                     tiles = 'ESRI',
                                                     title = 'Total translational displacement to date (mm)',
                                                     fontscale=0.95,
                                                     crs=crs).opts(opts.VectorField(magnitude=dim('T')*0.2,
                                                                                    rescale_lengths=False,
                                                                                    color='T',
                                                                                    colorbar=True,
                                                                                    colorbar_opts={'title': 'Total translational displacement (mm)'},
                                                                                   ))


    map_total_T = vectors_total_T * labels_T

    vectors_diff_T = ds_diff_mm.hvplot.vectorfield(x='NADE',
                                                   y='NADN',
                                                   angle='angle',
                                                   mag='T',
                                                   by='Point',
                                                   geo=True,
                                                   tiles = 'ESRI',
                                                   title = 'Rate of translational displacement (mm/day)',
                                                   fontscale=0.95,
                                                   crs=crs).opts(opts.VectorField(magnitude=dim('T')*0.2,
                                                                                  rescale_lengths=False,
                                                                                  color='T',
                                                                                  colorbar=True,
                                                                                  colorbar_opts={'title': 'Rate of translational displacement (mm/day)'},
                                                                                 ))


    map_diff_T = vectors_diff_T * labels_T

    points_total_Z = gdf_total_mm.hvplot(title='Total vertical displacement (mm)   pos=heave',
                                         geo=True,
                                         tiles='ESRI',
                                         groupby='Date',
                                         fontscale=0.95,
                                         c='Z').opts(opts.Points(colorbar=True,
                                                                 colorbar_opts={'title':'Total vertical displacement (mm)   pos=heave'},
                                                                 cmap='viridis',
                                                                 size=10))
    map_total_Z = points_total_Z * labels_Z

    points_diff_Z = gdf_diff_mm.hvplot(title='Rate of vertical displacement (mm/day)   pos=heave',
                                       geo=True,
                                       tiles='ESRI',
                                       groupby='Date',
                                       fontscale=0.95,
                                       c='Z').opts(opts.Points(colorbar=True,
                                                               colorbar_opts={'title':'Rate of vertical displacement (mm/day)   pos=heave'},
                                                               cmap='viridis',
                                                               size=10))
    map_diff_Z = points_diff_Z * labels_Z

    piezo_points = gdf_piezo_nad.assign(x=lambda gdf: gdf.geometry.x,
                                        y=lambda gdf: gdf.geometry.y).hvplot.points(x='x',
                                                                                    y='y',
                                                                                    geo=True)
    piezo_labels = gdf_piezo_nad.assign(x=lambda gdf: gdf.geometry.x,
                                        y=lambda gdf: gdf.geometry.y).hvplot.labels(x='x',
                                                                                    y='y',
                                                                                    text='Point',
                                                                                    geo=True,
                                                                                   ).opts(opts.Labels(text_font_size='9pt',
                                                                                                               xoffset=5,
                                                                                                               yoffset=5))
    piezo_map = infilling_extents_plot * piezo_points * piezo_labels

    condition = df_bh202['VWid'].isin(['VW1_excesspp', 'VW2_excesspp'])
    max_y = df_bh202.loc[condition, 'Head'].max()
    min_y = df_bh202.loc[condition, 'Head'].min()

    bh202_graph = bh202_pipeline.hvplot(x = 'Date',
                                        by='VWid',
                                        y='Head',
                                        ylabel='Pore pressure head (m)',
                                        line_width=1,
                                        title='BH 202: Total pore pressure head, river water level, and approx. excess pore pressure - VW1 13.3m [-10.15m EL] silty clay, VW2 23.9m [-20.75m EL] silty clay',
                                        legend=True,
                                        fontscale=0.95,
                                        grid=True,
                                        shared_axes=False
                                       )
    ma_bh202_VW1 = idf_bh202[idf_bh202['VWid'] == 'VW1'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()
    ma_bh202_VW2 = idf_bh202[idf_bh202['VWid'] == 'VW2'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()
    ma_bh202_VW1_ex = idf_bh202[idf_bh202['VWid'] == 'VW1_excesspp'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()
    ma_bh202_VW2_ex = idf_bh202[idf_bh202['VWid'] == 'VW2_excesspp'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()
    ma_bh202_River = idf_bh202[idf_bh202['VWid'] == 'River'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()

    bh202_graph_plot = bh202_graph * ma_bh202_VW1.hvplot(x='Date', label='VW1 24hr moving ave.') * ma_bh202_VW2.hvplot(x='Date', label='VW2 24hr moving ave.') * ma_bh202_VW1_ex.hvplot(x='Date', label='VW1 excessPP 24hr moving ave.') * ma_bh202_VW2_ex.hvplot(x='Date', label='VW2 excessPP 24hr moving ave.') * ma_bh202_River.hvplot(x='Date', label='River 24hr moving ave.')
    bh202_graph_plot = bh202_graph_plot.opts(ylim=(-2, 5), shared_axes=False)

    condition = df_bh204['VWid'].isin(['VW1_excesspp', 'VW2_excesspp'])
    max_y = df_bh204.loc[condition, 'Head'].max()
    min_y = df_bh204.loc[condition, 'Head'].min()

    bh204_graph = bh204_pipeline.hvplot(x = 'Date',
                                       by='VWid',
                                       y='Head',
                                       ylabel='Pore pressure head (m)',
                                       line_width=1,
                                       title='BH 204: Total pore pressure head, river water level, and approx. excess pore pressure - VW1 10.7m [-9.09m EL] silty clay, VW2 16.5m [-14.89m EL] silty clay',
                                       legend=True,
                                       fontscale=0.95,
                                       grid=True,
                                       shared_axes=False
                                      )
    ma_bh204_VW1 = df_bh204[df_bh204['VWid'] == 'VW1'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()
    ma_bh204_VW2 = df_bh204[df_bh204['VWid'] == 'VW2'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()
    ma_bh204_VW1_ex = df_bh204[df_bh204['VWid'] == 'VW1_excesspp'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()
    ma_bh204_VW2_ex = df_bh204[df_bh204['VWid'] == 'VW2_excesspp'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()
    ma_bh204_River = df_bh204[df_bh204['VWid'] == 'River'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()

    bh204_graph_plot = bh204_graph * ma_bh204_VW1.hvplot(x='Date', label='VW1 24hr moving ave.') * ma_bh204_VW2.hvplot(x='Date', label='VW2 24hr moving ave.') * ma_bh204_VW1_ex.hvplot(x='Date', label='VW1 excessPP 24hr moving ave.') * ma_bh204_VW2_ex.hvplot(x='Date', label='VW2 excessPP 24hr moving ave.') * ma_bh204_River.hvplot(x='Date', label='River 24hr moving ave.')
    bh204_graph_plot = bh204_graph_plot.opts(ylim=(-2, 5), shared_axes=False)

    condition = df_bh02['VWid'].isin(['VW1_excesspp', 'VW2_excesspp'])
    max_y = df_bh02.loc[condition, 'Head'].max()
    min_y = df_bh02.loc[condition, 'Head'].min()

    bh02_graph = bh02_pipeline.hvplot(x = 'Date',
                                       by='VWid',
                                       y='Head',
                                       ylabel='Pore pressure head (m)',
                                       line_width=1,
                                       title='BH-02: Total pore pressure head, river water level, and approx. excess pore pressure - VW1 6.43m [-5.13m EL] woodwaste',
                                       legend=True,
                                       fontscale=0.95,
                                       grid=True,
                                       shared_axes=False
                                      )
    ma_bh02_VW1 = df_bh02[df_bh02['VWid'] == 'VW1'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()
    ma_bh02_VW1_ex = df_bh02[df_bh02['VWid'] == 'VW1_excesspp'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()
    ma_bh02_River = df_bh02[df_bh02['VWid'] == 'River'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()

    bh02_graph_plot = bh02_graph * ma_bh02_VW1.hvplot(x='Date', label='VW1 24hr moving ave.') * ma_bh02_VW1_ex.hvplot(x='Date', label='VW1 excessPP 24hr moving ave.') * ma_bh02_River.hvplot(x='Date', label='River 24hr moving ave.')
    bh02_graph_plot = bh02_graph_plot.opts(ylim=(-2, 5), shared_axes=False)

    condition = df_bh03['VWid'].isin(['VW1_excesspp', 'VW2_excesspp'])
    max_y = df_bh03.loc[condition, 'Head'].max()
    min_y = df_bh03.loc[condition, 'Head'].min()

    bh03_graph = bh03_pipeline.hvplot(x = 'Date',
                                       by='VWid',
                                       y='Head',
                                       ylabel='Pore pressure head (m)',
                                       line_width=1,
                                       title='BH-03: Total pore pressure head, river water level, and approx. excess pore pressure - VW1 5.33m [-3.78m EL] rockfill, VW2 10.06m [-8.51m EL] woodwaste',
                                       legend=True,
                                       fontscale=0.95,
                                       grid=True,
                                       shared_axes=False
                                      )
    ma_bh03_VW1 = df_bh03[df_bh03['VWid'] == 'VW1'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()
    ma_bh03_VW2 = df_bh03[df_bh03['VWid'] == 'VW2'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()
    ma_bh03_VW1_ex = df_bh03[df_bh03['VWid'] == 'VW1_excesspp'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()
    ma_bh03_VW2_ex = df_bh03[df_bh03['VWid'] == 'VW2_excesspp'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()
    ma_bh03_River = df_bh03[df_bh03['VWid'] == 'River'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()

    bh03_graph_plot = bh03_graph * ma_bh03_VW1.hvplot(x='Date', label='VW1 24hr moving ave.') * ma_bh03_VW2.hvplot(x='Date', label='VW2 24hr moving ave.') * ma_bh03_VW1_ex.hvplot(x='Date', label='VW1 excessPP 24hr moving ave.') * ma_bh03_VW2_ex.hvplot(x='Date', label='VW2 excessPP 24hr moving ave.') * ma_bh03_River.hvplot(x='Date', label='River 24hr moving ave.')
    bh03_graph_plot = bh03_graph_plot.opts(ylim=(-2, 5), shared_axes=False)

    condition = df_bh05['VWid'].isin(['VW1_excesspp', 'VW2_excesspp'])
    max_y = df_bh05.loc[condition, 'Head'].max()
    min_y = df_bh05.loc[condition, 'Head'].min()

    bh05_graph = bh05_pipeline.hvplot(x = 'Date',
                                       by='VWid',
                                       y='Head',
                                       ylabel='Pore pressure head (m)',
                                       line_width=1,
                                       title='BH-05: Total pore pressure head, river water level, and approx. excess pore pressure - VW1 6.63m [-5.37m EL] woodwaste, VW2 9.85m [-8.59m EL] transitional',
                                       legend=True,
                                       fontscale=0.95,
                                       grid=True,
                                       shared_axes=False
                                      )
    ma_bh05_VW1 = df_bh05[df_bh05['VWid'] == 'VW1'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()
    ma_bh05_VW2 = df_bh05[df_bh05['VWid'] == 'VW2'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()
    ma_bh05_VW1_ex = df_bh05[df_bh05['VWid'] == 'VW1_excesspp'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()
    ma_bh05_VW2_ex = df_bh05[df_bh05['VWid'] == 'VW2_excesspp'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()
    ma_bh05_River = df_bh05[df_bh05['VWid'] == 'River'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()

    bh05_graph_plot = bh05_graph * ma_bh05_VW1.hvplot(x='Date', label='VW1 24hr moving ave.') * ma_bh05_VW2.hvplot(x='Date', label='VW2 24hr moving ave.') * ma_bh05_VW1_ex.hvplot(x='Date', label='VW1 excessPP 24hr moving ave.') * ma_bh05_VW2_ex.hvplot(x='Date', label='VW2 excessPP 24hr moving ave.') * ma_bh05_River.hvplot(x='Date', label='River 24hr moving ave.')
    bh05_graph_plot = bh05_graph_plot.opts(ylim=(-2, 5), shared_axes=False)

    condition = df_bh06['VWid'].isin(['VW1_excesspp', 'VW2_excesspp'])
    max_y = df_bh06.loc[condition, 'Head'].max()
    min_y = df_bh06.loc[condition, 'Head'].min()

    bh06_graph = bh06_pipeline.hvplot(x = 'Date',
                                       by='VWid',
                                       y='Head',
                                       ylabel='Pore pressure head (m)',
                                       line_width=1,
                                       title='BH-06: Total pore pressure head, river water level, and approx. excess pore pressure - VW1 9.37m [-6.42m EL] woodwaste',
                                       legend=True,
                                       fontscale=0.95,
                                       grid=True,
                                       shared_axes=False
                                      )
    ma_bh06_VW1 = df_bh06[df_bh06['VWid'] == 'VW1'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()
    ma_bh06_VW1_ex = df_bh06[df_bh06['VWid'] == 'VW1_excesspp'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()
    ma_bh06_River = df_bh06[df_bh06['VWid'] == 'River'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()

    bh06_graph_plot = bh06_graph * ma_bh06_VW1.hvplot(x='Date', label='VW1 24hr moving ave.') * ma_bh06_VW1_ex.hvplot(x='Date', label='VW1 excessPP 24hr moving ave.') * ma_bh06_River.hvplot(x='Date', label='River 24hr moving ave.')
    bh06_graph_plot = bh06_graph_plot.opts(ylim=(-2, 5), shared_axes=False)

    condition = df_bh10['VWid'].isin(['VW1_excesspp', 'VW2_excesspp'])
    max_y = df_bh10.loc[condition, 'Head'].max()
    min_y = df_bh10.loc[condition, 'Head'].min()

    bh10_graph = bh10_pipeline.hvplot(x = 'Date',
                                       by='VWid',
                                       y='Head',
                                       ylabel='Pore pressure head (m)',
                                       line_width=1,
                                       title='BH-10: Total pore pressure head, river water level, and approx. excess pore pressure - VW1 9.20m [-7.93m EL] woodwaste',
                                       legend=True,
                                       fontscale=0.95,
                                       grid=True,
                                       shared_axes=False
                                      )
    ma_bh10_VW1 = df_bh10[df_bh10['VWid'] == 'VW1'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()
    ma_bh10_VW1_ex = df_bh10[df_bh10['VWid'] == 'VW1_excesspp'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()
    ma_bh10_River = df_bh10[df_bh10['VWid'] == 'River'].rolling(window=144, on='Date', center=True).mean(numeric_only=True).dropna()

    bh10_graph_plot = bh10_graph * ma_bh10_VW1.hvplot(x='Date', label='VW1 24hr moving ave.') * ma_bh10_VW1_ex.hvplot(x='Date', label='VW1 excessPP 24hr moving ave.') * ma_bh10_River.hvplot(x='Date', label='River 24hr moving ave.')
    bh10_graph_plot = bh10_graph_plot.opts(ylim=(-2, 5), shared_axes=False)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Creating widgets and plots time elapsed: {elapsed_time:.2f} seconds")
    
    return point_button, diff_plot_T, total_plot_T, diff_plot_Z, total_plot_Z, map_diff_T, map_total_T, map_diff_Z, map_total_Z, piezo_map, bh204_graph_plot, bh02_graph_plot, bh10_graph_plot, bh03_graph_plot, bh05_graph_plot, bh06_graph_plot, bh202_graph_plot


# In[5]:


def update_template(point_button, diff_plot_T, total_plot_T, diff_plot_Z, total_plot_Z, map_diff_T, map_total_T, map_diff_Z, map_total_Z, piezo_map, bh204_graph_plot, bh02_graph_plot, bh10_graph_plot, bh03_graph_plot, bh05_graph_plot, bh06_graph_plot, bh202_graph_plot):
    #Layout using template
    template = pn.template.FastListTemplate(
        logo=r'./files/Hatch_Logo_Colour_RGB.png',
        title='H368314 - IPPL - Lee Cove Infilling - Monitoring Dashboard',
        #sidebar=[pn.pane.Markdown("## Settings for displacement graphs"), lower_date_slider, upper_date_slider, pn.pane.Markdown("## Settings for piezo graphs"), piezo_date_slider],
        main=[pn.Row('### Point Selection:', point_button),
              pn.Row(pn.Column(diff_plot_T.panel(height=400, width = 650), margin=(0,25)), total_plot_T.panel(height=400, width = 650)),
              pn.Row(pn.Column(diff_plot_Z.panel(height=400, width = 650), margin=(0,25)), total_plot_Z.panel(height=400, width = 650)),
              pn.Row(pn.Column(map_diff_T)),
              pn.Row(pn.Column(map_total_T)),
              pn.Row(pn.Column(map_diff_Z)),
              pn.Row(pn.Column(map_total_Z)),
              pn.Row(pn.Column(piezo_map)),
              pn.Row(pn.Column(bh204_graph_plot.panel(height=400, width = 1300))),
              pn.Row(pn.Column(bh02_graph_plot.panel(height=400, width = 1300))),
              pn.Row(pn.Column(bh10_graph_plot.panel(height=400, width = 1300))),
              pn.Row(pn.Column(bh03_graph_plot.panel(height=400, width = 1300))),
              pn.Row(pn.Column(bh05_graph_plot.panel(height=400, width = 1300))),
              pn.Row(pn.Column(bh06_graph_plot.panel(height=400, width = 1300))),
              pn.Row(pn.Column(bh202_graph_plot.panel(height=400, width = 1300))),
             ],
    )
    
    return template


# In[6]:


def update_dashboard():
    global template
    df_water, df_raw, df_nad, df_piezo_nad, df_bh202, df_bh204, df_bh02, df_bh03, df_bh05, df_bh06, df_bh10 = fetch_data()
    df_water, df_raw, df_nad, df_piezo_nad, df_bh202, df_bh204, df_bh02, df_bh03, df_bh05, df_bh06, df_bh10, df_total_mm, idf_diff_mm, idf_total_mm, idf_bh202, idf_bh204, idf_bh02, idf_bh03, idf_bh05, idf_bh06, idf_bh10, gdf_nad, ds_total_mm, ds_diff_mm, gdf_total_mm, gdf_diff_mm, gdf_piezo_nad, crs = process_data(df_water, df_raw, df_nad, df_piezo_nad, df_bh202, df_bh204, df_bh02, df_bh03, df_bh05, df_bh06, df_bh10)
    point_button, diff_plot_T, total_plot_T, diff_plot_Z, total_plot_Z, map_diff_T, map_total_T, map_diff_Z, map_total_Z, piezo_map, bh204_graph_plot, bh02_graph_plot, bh10_graph_plot, bh03_graph_plot, bh05_graph_plot, bh06_graph_plot, bh202_graph_plot = create_widgets_plots(df_water, df_raw, df_nad, df_piezo_nad, df_bh202, df_bh204, df_bh02, df_bh03, df_bh05, df_bh06, df_bh10, df_total_mm, idf_diff_mm, idf_total_mm, idf_bh202, idf_bh204, idf_bh02, idf_bh03, idf_bh05, idf_bh06, idf_bh10, gdf_nad, ds_total_mm, ds_diff_mm, gdf_total_mm, gdf_diff_mm, gdf_piezo_nad, crs)
    template = update_template(point_button, diff_plot_T, total_plot_T, diff_plot_Z, total_plot_Z, map_diff_T, map_total_T, map_diff_Z, map_total_Z, piezo_map, bh204_graph_plot, bh02_graph_plot, bh10_graph_plot, bh03_graph_plot, bh05_graph_plot, bh06_graph_plot, bh202_graph_plot)


# In[10]:


update_dashboard()
pn.state.add_periodic_callback(update_dashboard, period=1, count=1)

start_time = time.time()

# Disable server-side caching
os.environ["BOKEH_MINIFIED"] = "false"
os.environ["BOKEH_LOG_LEVEL"] = "debug"
os.environ["BOKEH_RESOURCES"] = "server"

#template.servable();
#template.show()

if __name__ == '__main__':
    pn.serve(template, address='0.0.0.0', port=int(os.environ.get('PORT', 5000)), allow_websocket_origin=["h368314-monitoring-dashboard.herokuapp.com"])


end_time = time.time()
elapsed_time = end_time - start_time

print(f"Creating dashboard time elapsed: {elapsed_time:.2f} seconds")


# In[8]:


# Clean up global variables
#for var_name in list(globals().keys()):
    #if var_name.startswith('df'):
        #del globals()[var_name]


# In[9]:


#import gc

# Force garbage collection
#gc.collect()


# In[ ]:




