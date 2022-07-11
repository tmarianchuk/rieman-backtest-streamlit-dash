from datetime import datetime
import json
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import connect, Error
from sshtunnel import SSHTunnelForwarder
import config
from sqlalchemy import create_engine
import time
import random
import requests
from random import sample
import math
import os
from dateutil.relativedelta import relativedelta
import sys

def open_connection(q):
    with open("ssh_config.txt", "r") as f:
        lines = f.readlines()
        hostname = lines[0].strip()
        username = lines[1].strip()
        password = lines[2].strip()
        remote_bind_address = lines[3].strip()
        dbname = lines[4].strip()
        postgres_pass = lines[5].strip()

    try:
        with SSHTunnelForwarder(
            (hostname, 22),
            ssh_username=username,
            ssh_password=password,
            remote_bind_address=(remote_bind_address, 5432),
            local_bind_address=("localhost", random.randint(6000,6300))) \
                as tunnel:

            tunnel.start()
            print("SSH connected.")

            engine = create_engine("postgresql+psycopg2://{}:{}@localhost:{}/{}".format(username,postgres_pass,tunnel.local_bind_port,dbname))
            batch_no = 0
            chunk_size = 5000
            df = []
            with engine.connect().execution_options(stream_results=True) as conn:
                conn.dialect.server_side_cursors = True
                print('db connection established')
                start = time.time()
                print('running the query')
                #tmp = pd.read_sql_query(q, conn)
                for chunk in pd.read_sql_query(q, conn,chunksize=chunk_size,coerce_float=False):
                    df.append(chunk)
                    #chunk.to_csv("./df_{:04d}.csv".format(batch_no),index=False)
                    batch_no += 1
                    print('chunk {} finished...'.format(batch_no))
                end = time.time()
                df = pd.concat(df)
                #print('converting to csv...')
                #df.to_csv('./df_full.csv',index=False)
                tot_time = round(end-start,2)
                if tot_time < 120:
                    print('query took {} seconds'.format(tot_time))
                else:
                    print('query took {} minutes'.format(tot_time/60))
                #tmp.to_csv("./df.csv",index=False)

            engine.dispose()
            tunnel.close()
            print("DB disconnected.")

            #df = pd.read_csv('./df.csv')
            print('query now acccessible as pandas df')
            return df
            #print('query now acccessible as pandas dict')#df')
            #return(df.to_dict(orient='records')[0])#(df)
    except (Exception, Error) as error:
        print("Error while connecting to DB", error)

def get_def_lps_for_state(state:str):
    state= state.upper()
    if state == 'IL':
        def_lps = ('C23')
    if state == 'PA':
        def_lps = ('RS', 'RG', 'RS-GRS', 'RSNH', 'R111')
    if state == 'OH':
        def_lps = ('RES', 'RG', 'RS00', 'RS0')
    if state == 'MA':
        def_lps = ('R1', 'R-1')
    if state == 'NY':
        def_lps = ('SC-1', 'SC1 (R)', 'SC-1.2')
    if state == 'MD':
        def_lps = ('RDNS', 'R', 'MDWOWH', 'RMNS')
    if state == 'NJ':
        def_lps = ('RS', 'RSNH')
    return def_lps

# SELECT BACKTEST DATE RANGE. CHANGE THIS IF DESIRED
start_date = '2021-01-01'
end_date = '2021-03-31'
zsp = pd.read_parquet('zip_sz_lp.parquet')
st = 'MD'#f'{sys.argv[0]}'
def_lps = get_def_lps_for_state(st)

STATE_BACKTEST_FILENAME = f'curated_{st}_backtest_for_enrolls_btw_{start_date}_and_{end_date}.csv'

if not os.path.exists(STATE_BACKTEST_FILENAME):
     if st == 'IL' or st == 'DC':
         curated_backtest_query= f'''with address_info as(select distinct on (utility_account_number)
                    utility_account_number, transmissionequivalent,peakloadcapacity,
                    svcaddress1, svccity, svcstate, svczip,
                    service_zone, utilityloadprofile
                    from smarts.meter_status),

         final as(select
         ai.utility_account_number,svcaddress1, svccity, svcstate, svczip,
         pv.iso_zone, pv.service_zone, pv.load_profile,
         pv.sales_channel,
         date_trunc('month',pv.service_start_month)::date as service_start_month,
         rbc.rce,
         transmissionequivalent as nits,
         peakloadcapacity as icap,
         pv.rate_type,
         pv.contract_rate, pv.netrate,
         pv.term, pv.enroll_on,
         pv.usage,
         pv.cost,
         pv.cost*pv.usage as total_cost, pv.total_charges, pv.margin
         from ltv.present_value pv
         left join address_info ai on ai.utility_account_number = pv.utility_account_number and ai.service_zone = pv.service_zone
         left join rce_by_customer rbc on rbc.utility_account_number = pv.utility_account_number and rbc.service_zone = pv.service_zone
         where svcstate='{st}' and load_profile in ('{def_lps}')
         and rce between 0.5 and 5
         and term in (3,6,12,24,36)
         and rate_type = 'CONTRACT'
         and sales_channel != 'Aggregation'
         and enroll_on between '{start_date}' and '{end_date}'
         order by enroll_on, utility_account_number, service_start_date)
    
         select *
         from final
         '''
     else:
         curated_backtest_query= f'''with address_info as(select distinct on (utility_account_number)
                    utility_account_number, transmissionequivalent,peakloadcapacity,
                    svcaddress1, svccity, svcstate, svczip,
                    service_zone, utilityloadprofile
                    from smarts.meter_status),

         final as(select
         ai.utility_account_number,svcaddress1, svccity, svcstate, svczip,
         pv.iso_zone, pv.service_zone, pv.load_profile,
         pv.sales_channel,
         date_trunc('month',pv.service_start_month)::date as service_start_month,
         rbc.rce,
         transmissionequivalent as nits,
         peakloadcapacity as icap,
         pv.rate_type,
         pv.contract_rate, pv.netrate,
         pv.term, pv.enroll_on,
         pv.usage,
         pv.cost,
         pv.cost*pv.usage as total_cost, pv.total_charges, pv.margin
         from ltv.present_value pv
         left join address_info ai on ai.utility_account_number = pv.utility_account_number and ai.service_zone = pv.service_zone
         left join rce_by_customer rbc on rbc.utility_account_number = pv.utility_account_number and rbc.service_zone = pv.service_zone
         where svcstate = '{st}' and load_profile in {def_lps}
         and rce between 0.5 and 5
         and term in (3,6,12,24,36)
         and rate_type = 'CONTRACT'
         and sales_channel != 'Aggregation'
         and enroll_on between '{start_date}' and '{end_date}'
         order by enroll_on, utility_account_number, service_start_month)
    
         select *
         from final
         '''

     curated_backtest_query.replace('\n',' ')
     # CREATE CURATED_BACKTEST
     curated_backtest = open_connection(curated_backtest_query)
     curated_backtest.nits.fillna(0,inplace=True)
     curated_backtest.icap.fillna(0,inplace=True)
     curated_backtest["netrate"].fillna(curated_backtest["contract_rate"], inplace=True)
     # SAVE CURATED_BACKTEST AS CSV
     curated_backtest.to_csv(f'curated_{st}_backtest_for_enrolls_btw_{start_date}_and_{end_date}.csv', index=False)
     print("finished creating backtesting data for curated accounts")

#if curated backtest already exists
if os.path.exists(STATE_BACKTEST_FILENAME):
     curated_backtest = pd.read_csv(STATE_BACKTEST_FILENAME)
     curated_backtest['service_start_month'] = curated_backtest['service_start_month'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d').date())

min_date = start_date #datetime.strftime(curated_backtest.service_start_month.min(),'%Y-%m-%d')
max_date = datetime.strftime(datetime.strptime(min_date,'%Y-%m-%d') + relativedelta(months=48),'%Y-%m-%d')
BACKTEST_BASE_FILENAME = f'cmv2_backtest_base_from_{min_date}_to_{max_date}.csv'

if not os.path.exists(BACKTEST_BASE_FILENAME):
    cmv2_defaults_4_backtest = f'''
    with cmv2_base as (
      select distinct on(service_zone, schedule_code, zone_name,month)
      schedule_code,
      zone_name,
      utility_short_name,
      service_zone,
      right(service_zone,2) as state,
      month,
      monthly_usage_kwh,
      round(cost_at_meter - capacity_per_mwh - transmission_per_mwh - arr_per_mwh,2) as cost_base,
      capacity_per_mwh,
      transmission_per_mwh,
      arr_per_mwh,
      default_icap_kw, default_nits_kw,
      (1 + interestrate * 30 / 365) / (1 - por - gross_receipts_tax - muni_excise) as mac,
      round(gross_cost,2) as gross_cost
      from costing_model_v2
      where month BETWEEN '{min_date}' and '{max_date}'
      and service_zone != 'nyseg_agg_ny'
    )

    select distinct on (cm.service_zone, schedule_code, zone_name,month)
    cm.service_zone,
    schedule_code as load_profile,
    zone_name,
    utility_short_name,
    state,
    month,
    monthly_usage_kwh,
    cost_base,
    capacity_per_mwh,
    transmission_per_mwh,
    arr_per_mwh,
    default_icap_kw,
    default_nits_kw,
    mac,
    gross_cost
    from cmv2_base cm
    order by cm.service_zone, schedule_code,zone_name,month
    '''

    cmv2_defaults_4_backtest.replace('\n',' ')
    # CREATE BACKTEST_BASE
    backtest_base = open_connection(cmv2_defaults_4_backtest)
    backtest_base['service_start_month'] = backtest_base.month.apply(lambda x: x.replace(day=1))
    # SAVE BACKTEST_BASE AS CSV
    backtest_base.to_csv(f'cmv2_backtest_base_from_{min_date}_to_{max_date}.csv', index=False)

    print("finished creating backtest base")

if os.path.exists(BACKTEST_BASE_FILENAME):
     #if backtest within date range already exists
     backtest_base = pd.read_csv(BACKTEST_BASE_FILENAME)#f'cmv2_backtest_base_from_{min_date}_to_{max_date}.csv')
     #backtest_base['service_start_month'] = backtest_base.month.apply(lambda x: datetime.strptime(x,'%Y-%m-%d').replace(day=1).date())
     backtest_base['service_start_month'] = backtest_base.service_start_month.apply(lambda x: datetime.strptime(x,'%Y-%m-%d').date())

# FORMAT BACKTEST_BASE TO BE ABLE TO JOIN TO CURATED_BACKTEST
iz_map = np.load('iz_map.pkl',allow_pickle=True)
backtest_base['iso_zone'] = backtest_base.zone_name.map(iz_map)
window = 12 # want a 12 month window to compute the rce on
backtest_base['default_rce'] = round((backtest_base.groupby(['service_zone','iso_zone','load_profile',np.arange(len(backtest_base)) // window])['monthly_usage_kwh'].transform("sum")/10000).astype(float),3)

# merge backtest_base onto the curated backtest
#print(type(curated_backtest['service_start_month'].iloc[0]), type(backtest_base['service_start_month'].iloc[0]) )
merged = curated_backtest.merge(backtest_base,how='left',on=['service_zone','iso_zone','load_profile','service_start_month'])
merged = merged.dropna()
# df of the curated accts, 1 row per acct number
curated_accts_df = curated_backtest[curated_backtest.utility_account_number.isin(merged.utility_account_number.unique())].groupby(['utility_account_number','svcaddress1','svccity','svcstate','svczip'],sort=False).first().reset_index()
#curated_accts_df = curated_backtest.groupby('utility_account_number',sort=False).first().reset_index()
curated_accts_df.svczip = curated_accts_df.svczip.astype(str)
#print(curated_accts_df)
if st == 'MA' or st == 'NJ':
     curated_accts_df.svczip = curated_accts_df.svczip.apply(lambda x: x.zfill(5))
# need to remove entries whose zipcodes we have no info for
curated_accts_df = curated_accts_df[curated_accts_df.svczip.isin(zsp.Zip.unique())]
# need to remove entries whose iso_zone have no info in the backtest_base
#curated_accts_df = curated_accts_df[curated_accts_df.iso_zone.isin(backtest_base.iso_zone.unique())]

# CREATE DATA DICT/JSON OF JUST CURATED ADDRESSES TO PUT THRU THE API
def get_pred_bill_dets(df:pd.DataFrame):
    tot_num_accts = len(df)
    max_api_calls = 300
    tot_api_call_batches = math.ceil(tot_num_accts/max_api_calls)
    final_results = []
    start_time = time.time()

    a = 0
    b = max_api_calls
    st = df.svcstate.iloc[0]
    print(f"Predicting billing determinants for {st} accounts...")
    for batch in range(tot_api_call_batches):
        if batch != range(tot_api_call_batches)[-1]:
            print(f'{st} accounts {a}-{b}')
            print(json.dumps(df.groupby(['utility_account_number',
                                         'svcaddress1',
                                         'svccity',
                                         'svcstate',
                                         'svczip'],sort=False,as_index=False).first().rename(columns={"svcaddress1":"AddressLine1",
                                                                                           "svccity":"City",
                                                                                           "svczip":"PostalCode",
                                                                                           "svcstate":"State"})[["AddressLine1","City","PostalCode","State"]][a:b].to_dict('records')),file=open(f'{a}-{b}_{st}accts.txt','w'))

            with open(f'{a}-{b}_{st}accts.txt') as f:
                accts = f.read()
                #print(accts)

            url = "http://3.144.128.30/package-predict"
            payload = {"package":json.loads(accts), "api_token": "46dc8d1d8f611ef5069d996bdf58a8389af5e292f10546767d3f3c524aadb645"}
            headers = {'accept': 'application/json','Content-Type': 'application/json'}
            res = requests.post(url, json=payload, headers=headers)


            results = [{key:json.loads(res.text)[n][key] for key in ['icap','nits','rce']} for n in range(len(json.loads(res.text)))]
            final_results += results
            a += max_api_calls
            b += max_api_calls
            time.sleep(5)
        else:
            #a += max_api_calls
            b += (tot_num_accts%max_api_calls) - max_api_calls
            print(f'{st} accounts {a}-{b}')
            print(json.dumps(df.groupby(['utility_account_number',
                                         'svcaddress1',
                                         'svccity',
                                         'svcstate',
                                         'svczip'],sort=False,as_index=False).first().rename(columns={"svcaddress1":"AddressLine1",
                                                                                           "svccity":"City",
                                                                                           "svczip":"PostalCode",
                                                                                           "svcstate":"State"})[["AddressLine1","City","PostalCode","State"]][a:b].to_dict('records')),file=open(f'{a}-{b}_{st}accts.txt','w'))

            with open(f'{a}-{b}_{st}accts.txt') as f:
                accts = f.read()
                #print(accts)

            url = "http://3.144.128.30/package-predict"
            payload = {"package":json.loads(accts), "api_token": "46dc8d1d8f611ef5069d996bdf58a8389af5e292f10546767d3f3c524aadb645"}
            headers = {'accept': 'application/json','Content-Type': 'application/json'}
            res = requests.post(url, json=payload, headers=headers)


            results = [{key:json.loads(res.text)[n][key] for key in ['icap','nits','rce']} for n in range(len(json.loads(res.text)))]
            final_results += results
    print(f"Finished predicting billing determinants for all {st} accounts enrolled in the selected date range! Time to complete: {time.time() - start_time}s")
    return final_results

# CREATE THE DATAFRAMES OF CURATED ACCOUNTS WITH THEIR PREDICTED BILLING DETERMINANTS AND ASSOCIATED COSTING MODEL ROWS
def rename_keys(dict_, new_keys:list):
    """
     new_keys: type List(), must match length of dict_
    """

    # dict_ = {oldK: value}
    # d1={oldK:newK,} maps old keys to the new ones:  
    d1 = dict( zip( list(dict_.keys()), new_keys) )

    # d1{oldK} == new_key 
    return {d1[oldK]: value for oldK, value in dict_.items()}

import gc
gc.collect()

ACCTS_W_BILL_PREDS_FILENAME = f"{st}_pred_bill_dets.csv"
if not os.path.exists(ACCTS_W_BILL_PREDS_FILENAME):
     # get ALL accts predicted bill dets from API
     results = get_pred_bill_dets(curated_accts_df)
     # rename dictionary keys to include prefix pred_
     final_results = [rename_keys(results[n],['pred_icap','pred_nits','pred_rce']) for n in range(len(results))]
     # convert predicted billing determinants from dictionary to df so it can be joined to accts
     pred_bill_dets = pd.DataFrame.from_dict(final_results)
     # append billing determinants to each acct number
     curated_accts_w_preds = pd.concat([curated_accts_df.reset_index(), pred_bill_dets], axis=1)[['utility_account_number','pred_icap','pred_nits','pred_rce']]
     curated_accts_w_preds.to_csv(ACCTS_W_BILL_PREDS_FILENAME, index=False)
# if csv will accounts and bill preds exists then load in
if os.path.exists(ACCTS_W_BILL_PREDS_FILENAME):
     curated_accts_w_pred = pd.read_csv(f"{st}_pred_bill_dets.csv")

# this merge contains row count equal to how many rows of real bills we've collected
realrows_merged = curated_backtest.merge(curated_accts_w_preds, how='left',on='utility_account_number').dropna(axis='rows')
# now merge to backtest_base to then process to compute the predicted gross cost based on predicted bill dets which we'll then use to calc custom rate
realrows_wdefs = realrows_merged.merge(backtest_base, how='left', on=['service_zone','iso_zone','load_profile','service_start_month'])
# USE ABOVE DF FOR REAL MARGIN
# USE BELOW DF TO COMPUTE CUSTOM RATES AND THEN JOIN BACK ONTO REAL ROW TO GET WHAT MARGIN WOULD BE BASED ON THE CUSTOM RATES
# takes only the relevant columns from df of curated accts, 1 row per acct
df = curated_accts_df[['utility_account_number','svcaddress1','svccity','svcstate','svczip','service_zone','iso_zone','load_profile','service_start_month','term']]
# duplicates df rows to match length of term, i.e. term length = num rows for that acct
df2 = df.loc[np.repeat(df.index.values, df.term)].reset_index().drop(columns='index', axis = 1).copy()
# create column to contain running count of the number of rows per acct so this can be used to increment the service start month
df2['mo2add'] = df2.groupby('utility_account_number').cumcount()
df2['corrected_service_start_month'] = df2.apply(lambda x: x.service_start_month + relativedelta(months=x.mo2add), axis=1).reset_index().drop(columns=['index'])
df2.drop(columns=['mo2add','service_start_month'], axis=1, inplace=True)
df2.rename(columns={'corrected_service_start_month':'service_start_month'},inplace=True)
# this merge contains row count equal to term length
synthrows_merged=df2.merge(curated_accts_w_preds, how='left',on='utility_account_number').dropna(axis='rows')

# now merge to backtest_base to then process to compute the predicted gross cost based on predicted bill dets which we'll then use to calc custom rate
synthrows_wdefs = synthrows_merged.merge(backtest_base, how='left', on=['service_zone','iso_zone','load_profile','service_start_month'])

# COMPUTE GROSS COST FROM PRED BILL DETS AND CALCULATE RATES
def compute_gross_cost(tmp:pd.DataFrame):
    tmp[['monthly_usage_kwh', 'cost_base', 'capacity_per_mwh',
       'transmission_per_mwh', 'arr_per_mwh', 'default_icap_kw',
       'default_nits_kw', 'mac','rce','nits','icap']] = tmp[['monthly_usage_kwh', 'cost_base', 'capacity_per_mwh',
       'transmission_per_mwh', 'arr_per_mwh', 'default_icap_kw',
       'default_nits_kw', 'mac','rce','nits','icap']].apply(pd.to_numeric)
    if tmp.empty:
        print("no results/empty df")
    if tmp.transmission_per_mwh.isnull().sum() != 0:
        tmp['pred_gross_cost'] = round( tmp.mac * (tmp.default_rce/tmp.pred_rce) * (tmp.cost_base + (tmp.capacity_per_mwh * (tmp.pred_icap/tmp.default_icap_kw)) + (tmp.transmission_per_mwh * (tmp.pred_nits/tmp.default_nits_kw)) + (tmp.arr_per_mwh * (tmp.pred_nits/ tmp.default_nits_kw))),2 )
        tmp['computed_gross_cost'] = round( tmp.mac * (tmp.default_rce/tmp.rce) * (tmp.cost_base + (tmp.capacity_per_mwh * (tmp.icap/tmp.default_icap_kw)) + (tmp.transmission_per_mwh * (tmp.nits/tmp.default_nits_kw)) + (tmp.arr_per_mwh * (tmp.nits/ tmp.default_nits_kw))),2 )
    else:
        tmp['pred_gross_cost'] = round( tmp.mac * (tmp.default_rce/tmp.pred_rce) * (tmp.cost_base + (tmp.capacity_per_mwh * (tmp.pred_icap/tmp.default_icap_kw)) + (tmp.arr_per_mwh * tmp.pred_nits)),2 )
        tmp['computed_gross_cost'] = round( tmp.mac * (tmp.default_rce/tmp.rce) * (tmp.cost_base + (tmp.capacity_per_mwh * (tmp.icap/tmp.default_icap_kw).replace(np.nan, 0)) + (tmp.arr_per_mwh * tmp.nits)),2 )
    return tmp[['utility_account_number','term','month','monthly_usage_kwh','gross_cost','computed_gross_cost','pred_gross_cost','default_rce','rce','pred_rce','default_icap_kw','icap','pred_icap','default_nits_kw','nits','pred_nits','contract_rate','netrate']]


def compute_pred_gross_cost(tmp:pd.DataFrame):
    tmp[['monthly_usage_kwh', 'cost_base', 'capacity_per_mwh',
       'transmission_per_mwh', 'arr_per_mwh', 'default_icap_kw',
       'default_nits_kw', 'mac']] = tmp[['monthly_usage_kwh', 'cost_base', 'capacity_per_mwh',
       'transmission_per_mwh', 'arr_per_mwh', 'default_icap_kw',
       'default_nits_kw', 'mac']].apply(pd.to_numeric)
    if tmp.empty:
        print("no results/empty df")
    if tmp.transmission_per_mwh.isnull().sum() != 0:
        tmp['pred_gross_cost'] = round( tmp.mac * (tmp.default_rce/tmp.pred_rce) * (tmp.cost_base + (tmp.capacity_per_mwh * (tmp.pred_icap/tmp.default_icap_kw)) + (tmp.transmission_per_mwh * (tmp.pred_nits/tmp.default_nits_kw)) + (tmp.arr_per_mwh * (tmp.pred_nits/ tmp.default_nits_kw))),2 )
    else:
        tmp['pred_gross_cost'] = round( tmp.mac * (tmp.default_rce/tmp.pred_rce) * (tmp.cost_base + (tmp.capacity_per_mwh * (tmp.pred_icap/tmp.default_icap_kw)) + (tmp.arr_per_mwh * tmp.pred_nits)),2 )
    return tmp[['utility_account_number','term','month','monthly_usage_kwh','gross_cost','pred_gross_cost','default_rce','pred_rce','default_icap_kw','pred_icap','default_nits_kw','pred_nits']]

def compute_rates(df: pd.DataFrame):
    marg = 0.1
    if 'computed_gross_cost' in df.columns:
        compute_rate = df.groupby('utility_account_number', sort=False)[['pred_gross_cost',
                                                                         'pred_rce_scaled_usage',
                                                                         'computed_gross_cost',
                                                                         'rce_scaled_usage']].transform(lambda x: sum(x))
        compute_rate['model_rate'] = round((compute_rate.pred_gross_cost / compute_rate.pred_rce_scaled_usage)/(1-marg),3)
        compute_rate['computed_rate'] = round((compute_rate.computed_gross_cost / compute_rate.rce_scaled_usage)/(1-marg),3)
        computed_rate = pd.concat([
            df[['utility_account_number','month']],
            compute_rate
        ], axis=1).rename(columns={'pred_gross_cost':'sum_pred_gross_cost',
                                  'pred_rce_scaled_usage':'sum_pred_rce_scaled_usage',
                                  'computed_gross_cost':'sum_computed_gross_cost',
                                  'rce_scaled_usage':'sum_rce_scaled_usage'})
        computed_final_df = df.merge(computed_rate, how='left', on=['utility_account_number','month'])
        return computed_final_df
    else:
        compute_model_rate = df.groupby('utility_account_number', sort=False)[['pred_gross_cost','pred_rce_scaled_usage']].transform(lambda x: sum(x))
        compute_model_rate['model_rate'] = round((compute_model_rate.pred_gross_cost / compute_model_rate.pred_rce_scaled_usage)/(1-marg),3)
        computed_model_rate = pd.concat([
            df[['utility_account_number','month']],
            compute_model_rate
        ], axis=1).rename(columns={'pred_gross_cost':'sum_pred_gross_cost',
                                  'pred_rce_scaled_usage':'sum_pred_rce_scaled_usage'})
        model_final_df = df.merge(computed_model_rate, how='left', on=['utility_account_number','month'])
        return model_final_df

def compute_margins(df:pd.DataFrame):
    # Convert columns to numeric so they can have math done on them
    df[['pred_rce', 'default_rce',
        'model_rate','computed_rate',
        'pred_gross_cost','computed_gross_cost']] = df[['pred_rce', 'default_rce',
                                                        'model_rate','computed_rate',
                                                        'pred_gross_cost','computed_gross_cost']].apply(pd.to_numeric)
    # compute total charge and margin using the model rate
    df['model_tot_charge'] = df.pred_rce_scaled_usage*df.model_rate
    df['model_margin'] = df['model_tot_charge'] - df['pred_gross_cost']
    if 'computed_rate' in df:
        # compute total charge and margin using the computed rate
        df['computed_tot_charge'] = df.rce_scaled_usage*df.computed_rate
        df['computed_margin'] = df['computed_tot_charge'] - df['computed_gross_cost']
    return df

def calc_rate_adder(temp:pd.DataFrame, marg:float):
    w_adder = temp.groupby('utility_account_number', as_index=False).agg({"computed_gross_cost":"sum",
                                              "rce_scaled_usage":"sum",
                                              "model_rate":"mean"})
    w_adder['marg_based_rate'] = w_adder.computed_gross_cost/(w_adder.rce_scaled_usage*(1-marg))
    w_adder['model_rate_adder'] = w_adder.marg_based_rate - w_adder.model_rate
    return w_adder

# compute gross cost from predicted billing determinants
computed_cost = compute_pred_gross_cost(synthrows_wdefs)
computed_cost['pred_rce_scaled_usage'] = (computed_cost.pred_rce.astype(float)/computed_cost.default_rce.astype(float))*computed_cost.monthly_usage_kwh.astype(float)

# compute gross cost from predicted and real billing determinants
real_bill_dets = realrows_wdefs[['utility_account_number','rce','nits','icap','contract_rate','netrate']].drop_duplicates(subset='utility_account_number')
real_and_pred_full_terms_df = synthrows_wdefs.drop(columns=['svcaddress1','svccity','svcstate','svczip']).merge(real_bill_dets, how='left', on='utility_account_number')
cost_real_and_pred_full_term = compute_gross_cost(real_and_pred_full_terms_df)
# create columns for real rce scaled default usage and predicted rce scaled default usage
cost_real_and_pred_full_term['pred_rce_scaled_usage'] = (cost_real_and_pred_full_term.pred_rce.astype(float)/cost_real_and_pred_full_term.default_rce.astype(float))*cost_real_and_pred_full_term.monthly_usage_kwh.astype(float)
cost_real_and_pred_full_term['rce_scaled_usage'] = (cost_real_and_pred_full_term.rce.astype(float)/cost_real_and_pred_full_term.default_rce.astype(float))*cost_real_and_pred_full_term.monthly_usage_kwh.astype(float)

# df with rates computed from pred gross cost
cost_w_model_rates = compute_rates(computed_cost)
# df with rates computed from pred gross cost (which uses pred rce/nits/icap) and computed gross cost (which uses real rce/nits/icap)
model_cost_and_rates = compute_rates(cost_real_and_pred_full_term)
# compute margins from above modeled and computed costs & rates
with_margs = compute_margins(model_cost_and_rates)
# compute the rate adder for a selected margin
set_marg = 0.1 # 10% profit margin
with_rate_adder = calc_rate_adder(with_margs,set_marg)
with_rate_adder.to_csv(f"rate_adders_{100*set_marg}p_margin_{start_date}-{end_date}_{st}.csv", index=False)

print(f'computed average rate: {round(with_margs.computed_rate.mean(),4)} \nmodel average rate: {round(with_margs.model_rate.mean(),4)}\n')
print(f'computed total gross cost: {round(with_margs.computed_gross_cost.sum(),2)} \nmodel total gross cost: {round(with_margs.pred_gross_cost.sum(),2)}\n')
print(f'computed total charges: {round(with_margs.computed_tot_charge.sum(),2)} \nmodel total charges: {round(with_margs.model_tot_charge.sum(),2)}\n')
print(f'avg rate adder = computed total gross cost - model total charges(i.e. model revenue) / actual rce scaled default usage: {round((with_margs.computed_gross_cost.sum()-with_margs.model_tot_charge.sum())/with_margs.rce_scaled_usage.sum(),3)}\n')
print(f'computed margin: {round(with_margs.computed_margin.sum(),2)} \nmodel margin: {round(with_margs.model_margin.sum(),2)}\n')


# below df uses default usage and assumes full term, and then model rates from pred bill dets and pred gross cost and computed rates from real bill dets and "real" computed gross cost to compare what the total charges and margin would have been at the model rate/pred gross cost/default usage vs computed rate/computed cost/default usage (which i think just becomes a measure of the performance of the bill dets model?)
print('creating synthetic "real" rows vs model csv')
with_margs.to_csv(f"synthrealvmodel_{start_date}-{end_date}_{st}.csv", index=False)
print("FINISHED!\n")
# below df uses actual usage and bills tendered, and then model rates from pred bill dets and pred gross cost to compute what the total charges and margin would have been at the real cost and usage (which i think becomes a measure of the rieman pricing system as a whole since it's comparing margin from the rieman product vs the actual product)
keepers = ['utility_account_number','month',
           'contract_rate','netrate','cost','total_cost','total_charges','margin','usage',
           'monthly_usage_kwh_x','pred_rce_scaled_usage','model_rate','term',
          'rce','nits','icap','default_rce_x','pred_rce','pred_nits','pred_icap','pred_gross_cost']
# joining to merged to filter df down to just real rowd (thereby implicitly including the actual retention for each acct)
for_final_analysis = merged.merge(cost_w_model_rates, how='left', on=['utility_account_number','term','month']).dropna(axis='rows')[keepers]

# make a column of actual usage * model rate to see what the total charge would be
for_final_analysis['model_tot_charge'] = for_final_analysis.usage.astype(float)*for_final_analysis.model_rate
# make a column of modeled total charge - total cost to see what the model margin is
for_final_analysis['model_margin'] = for_final_analysis['model_tot_charge'].astype(float) - for_final_analysis['total_cost']

# model margin may be negative because we are using the real usage and cost not predicted values
print(f'actual average netrate: ${round(for_final_analysis.netrate.mean(),4)}/kwh \nmodel average rate: ${round(for_final_analysis[for_final_analysis.model_rate!=for_final_analysis.model_rate.max()].model_rate.mean(),4)}/kwh\n')
print(f'actual total gross cost: ${round(for_final_analysis.total_cost.sum(),2)} \nmodel total gross cost: ${round(for_final_analysis[for_final_analysis.model_rate!=for_final_analysis.model_rate.max()].pred_gross_cost.sum(),2)}\n')
print(f'actual total charges: ${round(for_final_analysis.total_charges.sum(),2)} \nmodel total charges: ${round(for_final_analysis[for_final_analysis.model_rate!=for_final_analysis.model_rate.max()].model_tot_charge.sum(),2)}\n')
print(f'actual margin: ${round(for_final_analysis.margin.sum(),2)} \nmodel margin: ${round(for_final_analysis[for_final_analysis.model_rate!=for_final_analysis.model_rate.max()].model_margin.sum(),2)}\n')

print("creating real vs model csv")
for_final_analysis.to_csv(f"realvmodel_{start_date}-{end_date}_{st}.csv", index=False)
print("FINISHED!")
