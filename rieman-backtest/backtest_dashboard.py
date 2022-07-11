import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import streamlit as st
import psycopg2
from psycopg2 import connect, Error
from sshtunnel import SSHTunnelForwarder
import config
from sqlalchemy import create_engine
import time
import json
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
#import plotly.io as pio
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from millify import millify
import random
import matplotlib.cm as cm
# from lifelines import KaplanMeierFitter
# from lifelines.plotting import add_at_risk_counts
# from lifelines.utils import survival_table_from_events
from streamlit.elements.arrow import Data
import seaborn as sns

################### PAGE CONFIGURATION SETTINGS ##########################
# dashboard webtab visuals
st.set_page_config(page_title='rieman backtest', page_icon='✨', 
layout="wide", initial_sidebar_state="auto")

# dashboard page layout
option = st.sidebar.selectbox('Choose a page',('Main','Rieman Backtest','Contract Groove (Retention) Backtest'))
if (option != 'Main'):
    st.title(option)
else:
    st.title('Welcome to your data dashboard')

######################## LOADING IN PSQL QUERIES FROM DB #################################

# CREATE DATABASE CONNECTION
@st.cache(allow_output_mutation=True)
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
            local_bind_address=("localhost", random.randint(6000,6200))) \
                as tunnel:

            tunnel.start()
            print("SSH connected.")

            engine = create_engine("postgresql+psycopg2://{}:{}@localhost:{}/{}".format(username,postgres_pass,tunnel.local_bind_port,dbname))
            batch_no = 0
            chunk_size = 1000
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
                print('query took {} seconds'.format(round(end-start,2)))
                #tmp.to_csv("./df.csv",index=False)

            engine.dispose()
            tunnel.close()
            print("DB disconnected.")

            #df = pd.read_csv('./df.csv')
            print('query now acccessible as pandas df')
            return(df)
    except (Exception, Error) as error:
        print("Error while connecting to DB", error)

################# PSQL SNIPPET FILE LOAD IN ######################
snippet = './good_bills.sql'
with open(snippet) as f:
    goodbills = f.read().splitlines()
tmp = ''.join(goodbills)
##################################################################
# psql query standalone or built off snippet
query = ''' SELECT upper(right(service_zone,2)) as state_name,
            month as mobill,
            round(sum(net_sales_billed),2)::float as totbill, 
            round(sum(billable_usage),2)::float as totuse
            FROM allbills
            group by 1,2
            order by 1,2'''
query = query.replace('\n',' ')
#'''SELECT * FROM ltv.present_value LIMIT 10000'''
##################################################################
# join snippet and query
full_query = ''.join([tmp,query])
# load full_query into df

data = open_connection(full_query)
data['mobill'] = data['mobill'].astype(str)

#st.text(data.dtypes)
cpy_df = data.copy()
##### modifying multiselect colors by inserting html #####
# st.markdown("""
# <style>
# /* The input itself */
# div[data-baseweb="select"] > div {
#   /*background-color: orange !important;*/
#   font-size: 20px !important;
# }

# /* The list of choices */
# li>span {
#   color: white !important;
#   font-size: 20px;
#   /*background-color: blue !important;*/
# }

# li {
#   /*background-color: green !important;*/
# }
# </style>
# """, unsafe_allow_html=True)
######################################################################################

###################### FUNCS FOR VIEWING AND SAVING DFS ######################################
def view_df(df: pd.DataFrame):
    st.write('### View your dataframe:')
    all_cols = list(df.columns)
    all_cols.insert(0,'select all')
    cols = st.multiselect("Choose columns (you'll only be shown the first 15 rows)", all_cols)#df.columns.tolist(), default=default_cols) 
    if 'select all' in cols:
        cols = list(df.columns)#all_cols
    df = df[cols]
    #st.dataframe(df[cols].head(50))
    st.write(df.head(15))

# SAVING DF TO A CSV
def save_csv(df:pd.DataFrame):
    st.write('### Save your dataframe as a csv:')
    csv_fname = str(st.text_input("Enter filename to save dataframe as csv",max_chars=20))
    if('.csv' not in csv_fname and len(csv_fname)>0):
        csv_fname += '.csv'
    if st.button('save'):
        if len(csv_fname)>0:
            df.to_csv(csv_fname)
            st.markdown('### ✅ File saved!')
        else:
            st.markdown('### No filename given. Nothing saved 🤷')
######################################################################################

########################## FUNC FOR BY MONTH GEO PLOT ###############################
def plot_by_month(df: pd.DataFrame, date, date_label):
    tmp = df[df.mobill == str(date)]
    #st.write(tmp)
    mx = df.totbill.max()
    mn = df.totbill.min()
    
    fig = px.choropleth(tmp,  # Input Pandas DataFrame
                    locations=tmp.state_name,  # DataFrame column with locations
                    color="totbill",  # DataFrame column with color values
                    range_color=(mn,mx),
                    color_continuous_scale='dense',#'magenta',
                    center={'lat':40.42,'lon':-82.91}, #center of ohio
                    hover_name="totbill", # DataFrame column hover info
                    locationmode = 'USA-states', # Set to plot as US States
                    scope='usa') # Plot only the USA instead of globe
    fig.update_geos(fitbounds='locations') #zooms into center
    fig.add_scattergeo(locations=tmp.state_name, locationmode='USA-states',text=tmp.state_name, mode='text',
                        hoverinfo='none',textfont={'color':'black','size':20})
    fig.update_coloraxes(colorbar={'title':'Billed ($)','len':0.65, 'thickness':20})
    fig.update_layout( title={'text': 'State heatmap by total bill($) in {}'.format(date_label), 'font_size':25, 'x': 0.48, 'y':.95, 'xanchor': 'center'},
        #title_text = 'State Rankings by total bill($) in {}'.format(date_label), # Create a Title
        margin=dict(l=0, r=0, b=0, t=0, pad=0, autoexpand=True), height=400,
        hoverlabel={'font_size':18}
    )
    st.plotly_chart(fig, use_container_width=True)
######################################################################################

########################## FUNC FOR BY MONTH HIST PLOT ###############################
def by_month_hist(df:pd.DataFrame,date, date_label):
    fig = px.histogram(df[df['mobill']==date], x='state_name', y='totbill',
                        title='Histogram of bill totals for {}'.format(date_label), 
                        labels={'state_name':'state'})
    fig.update_layout(title_font_size=25, title_x=0.5, title_y=0.88,
    margin=dict(l=0, r=0, b=0, t=80, pad=0, autoexpand=True), height=360,
    hoverlabel={'font_size':18,'bgcolor':'dimgray','font_color':'white','bordercolor':'white'})
    cushion=100000
    fig.update_yaxes(title_text='total bill ($)',range=(0,df.totbill.max()+cushion))
    st.plotly_chart(fig, use_container_width=True)

def compute_monthly_tot(df:pd.DataFrame,date):
    chosen_mo_sum = df[df['mobill']==date].totbill.sum()
    prior_date = datetime.strptime(date,'%Y-%m-%d') - relativedelta(months=1)
    str_prior_date = datetime.strftime(prior_date,'%Y-%m-%d')
    prior_mo_sum = df[df['mobill']==str_prior_date].totbill.sum()
    diff = round(chosen_mo_sum-prior_mo_sum,2)
    return(chosen_mo_sum, diff)
######################################################################################

################## CODE PERTAINING TO MAIN PAGE LIES HERE #####################
if option == 'Main':
    #st.subheader('Here\'s how the monthly KPIs are looking')
    ##### plot a map with current customer count per state #####
    dates = data.mobill.unique()
    check = [datetime.strptime(x, '%Y-%m-%d') for x in dates] #converts string to datetime object
    #st.text(type(check[0])) # checking conversion
    
    date_slider = st.select_slider('DATES', options = check, format_func= lambda x: x.strftime("%m/%d/%Y"))
    date_str = date_slider.strftime("%Y-%m-%d")
    date_lbl = date_slider.strftime("%m/%d/%Y")

    map,hist = st.columns(2)
    with map:
        plot_by_month(data,date_str,date_lbl)

    with hist:
        by_month_hist(data,date_str,date_lbl)
        st.metric(label='Monthly total', 
        value=f"${millify(compute_monthly_tot(data,date_str)[0], precision=2)}",
        delta=millify(compute_monthly_tot(data,date_str)[1],precision=2) )
    
    #st.write(data[data.state_name == 'DC'])
    #############################################################

    # kpis, growth, retention = st.columns(3)
    # with kpis:
    #     st.subheader('KPIs')
    #     note = st.text_area('here we look at key performance indicators ')
    #     st.markdown(note)
    # with growth:
    #     st.subheader('Growth')
    #     note = st.text_area('here we look at marketing ')
    #     st.markdown(note)
    # with retention:
    #     st.subheader('Retention')
    #     note = st.text_area('here we look at metrics like clv, etc.')
    #     st.markdown(note)
################################################################################
# @st.cache(allow_output_mutation=True)
# def load_ltv():
#     ltv = pd.read_parquet('/Users/tmarianchuk/Desktop/csvs_but_smaller/ltv_present_value_11_17_2021.parquet')
#     ltv.rename(columns={'service_start_month':'service_start_date','total_charges':'revenue'},inplace=True)
#     ltv['state_name'] = ltv.service_zone.apply(lambda x: x.split('_', -1)[-1].upper())
#     no_nulls_ltv = ltv[~ltv.margin.isnull() & ~ltv.nits.isnull() & ~ltv.icap.isnull()].drop(columns='municipality')
#     no_nulls_ltv['cogs'] = no_nulls_ltv['cost'] * no_nulls_ltv['usage']
#     no_nulls_ltv['term'] = no_nulls_ltv['term'].astype(str)
#     return(ltv, no_nulls_ltv)
#@st.cache
def metric_bar_plot_grouped_by(col, othercol,target:str, df:pd.DataFrame,year):
    if (col == 'none') & (othercol == 'none'):
        st.write('### Please select a category, \'none\' is currently selected.')
    elif (col == 'none') | (othercol == 'none') | (col==othercol):
        if col == 'none':
            col = othercol
        elif othercol == 'none':
            col = col

        tmp = df[df.enroll_on >= year]
        tmp = pd.DataFrame(tmp.groupby(col)[target].sum().sort_values(ascending=False)).reset_index()
        tmp[target] = tmp[target].round(2)
        tmp['log_'+str(target)] = np.log(tmp[target])
        tmp['bins'] = pd.qcut(tmp[target],5)
        tmp['bin_labels'] = pd.qcut(tmp[target],5,labels=['lowest','low','mid','high','highest'])
        #margin_bins = [1000,10000,100000,1000000,10000000,100000000]
        #margin_labels=['lowest','low','mid','high','highest']
        #tmp['margin_bins'] = pd.cut(tmp['margin'], margin_bins, labels=margin_labels)
        #tmp['margin_strata'] = pd.cut(tmp['margin'], margin_bins)
        if tmp[target].min() > 0:
            fig = px.bar(tmp, x=col, y='log_'+str(target), log_y=True, text=target, color='bin_labels',
                        title=f"{target} vs. {col.replace('_',' ')} from {year}",
                        hover_data={col:True,target:True,'log_'+str(target):False,'bin_labels':False})#,facet_col='margin_bins')
            fig.update_layout(title={'font_size':30},legend_title_text='click to toggle group off:',
                            xaxis={'tickangle': 90, 'title_font_size': 20, 'title_text':col.replace('_',' '),
                                    'showticklabels': True,
                                    'type': 'category', 'categoryorder':'category ascending', 'linecolor':'white',
                                    'visible':True,'showgrid':False, 'showticklabels':True}, margin = {'pad': 20},
                            yaxis={'title_text':f'log({target})', 'title_font_size': 20,'visible':True,'showgrid':False, 'linecolor':'white'},
                            uniformtext_minsize=10,legend={'font_size':18,'groupclick':'togglegroup','itemdoubleclick':'toggleothers'})#,'itemclick':'toggleothers'})
            fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.bar(tmp, x=col, y=target, log_y=False, text=target, color='bin_labels',
                        title=f"{target} vs. {col.replace('_',' ')} from {year}",
                        hover_data={col:True,target:True,'log_'+str(target):False,'bin_labels':False})#,facet_col='margin_bins')
            fig.update_layout(title={'font_size':30},legend_title_text='click to toggle group off:', plot_bgcolor='rgba(0,0,0,0)',
                            xaxis={'tickangle': 90, 'title_font_size': 20, 'title_text':col.replace('_',' '),
                                    'showticklabels': True,
                                    'type': 'category', 'categoryorder':'category ascending', 'linecolor':'white',
                                    'visible':True,'showgrid':False, 'showticklabels':True}, margin = {'pad': 20},
                            yaxis={'title_text':f'{target}', 'title_font_size': 20,'visible':True,'showgrid':True, 'linecolor':'white'},
                            uniformtext_minsize=10,legend={'font_size':18, 'groupclick':'togglegroup','itemdoubleclick':'toggleothers'})#,'itemclick':'toggleothers'})
            fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        view_df(tmp)
        save_csv(tmp)
        #return(tmp)
    else:
        cols=[col,othercol]
        tmp = df[df.enroll_on >= year]
        tmp = pd.DataFrame(tmp.groupby(cols)[target].sum().sort_values(ascending=False)).reset_index()
        tmp[target] = tmp[target].round(2)
        tmp['log_'+str(target)] = np.log(tmp[target])
        tmp['bins'] = pd.qcut(tmp[target],5)
        tmp['bin_labels'] = pd.qcut(tmp[target],5,labels=['lowest','low','mid','high','highest'])
        
        fig = px.bar(tmp, x=cols[0], y=target, log_y=False, text=target,
                    color=cols[1],title=f"{target} vs. {cols[0].replace('_',' ')} by {cols[1].replace('_',' ')} from {year}",
                    hover_data={cols[0]:True,cols[1]:True,target:True,'log_'+str(target):False,'bin_labels':False})#,facet_col='margin_bins')
        fig.update_layout(title={'font_size':30}, xaxis={'tickangle': 90, 'title_font_size': 20, 'title_text':col.replace('_',' '),
                                'showticklabels': True,
                                'type': 'category', 'categoryorder':'category ascending', 'linecolor':'white',
                                    'visible':True,'showgrid':False, 'showticklabels':True},margin = {'pad': 20},
                        yaxis={'title_text':f'{target}', 'title_font_size': 20,'visible':True,'showgrid':True, 
                            'linecolor':'white'},
                        uniformtext_minsize=10,legend={'groupclick':'togglegroup','itemdoubleclick':'toggleothers'})
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        view_df(tmp)
        save_csv(tmp)
        #return(tmp)

def filterby_iqr(df:pd.DataFrame):
    dummy_df = df.copy()
    cols = ['margin', 'cogs', 'revenue','margin_percent'] # The columns you want to search for outliers in
    # Calculate quantiles and IQR
    Q1,Q3 = dummy_df[cols].quantile(0.25), dummy_df[cols].quantile(0.75) # Same as np.percentile but maps (0,1) and not (0,100)
    IQR = Q3 - Q1
    # Return a boolean array of the rows with (any) non-outlier column values
    condition = ~((dummy_df[cols] < (Q1 - 1.5 * IQR)) | (dummy_df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    # Filter our dataframe based on condition
    iqr_filtered_df = dummy_df[condition]
    return(iqr_filtered_df)

def create_quartiles(df:pd.DataFrame, col:str):
    df[str(col)+'_labels'] = pd.qcut(df[col],5,labels=['low '+str(col),'low-mid '+str(col),'mid '+str(col),'mid-high '+str(col),'high '+str(col)])
    df[str(col)+'_bins'] = pd.qcut(df[col],5)
    return(df)

def just_categoricals(df:pd.DataFrame):
    df = df.select_dtypes(['object','category']) 
    return(df)

def just_numericals(df:pd.DataFrame):
    df = df.select_dtypes(['int','float']) 
    return(df)

################################################################################
# def load_data():
#     my_bar = st.progress(0)
#     for percent_complete in range(100):
#         time.sleep(0.1)
#         my_bar.progress(percent_complete + 1)


# ------------------- RIEMAN BACKTEST -------------------- #

def calc_rate_adder(temp:pd.DataFrame, marg):
    w_adder = temp.groupby('utility_account_number', as_index=False).agg({"computed_gross_cost":"sum",
                                              "rce_scaled_usage":"sum",
                                              "model_rate":"mean",
                                              "term":"mean"})
    w_adder['marg_based_rate'] = w_adder.computed_gross_cost/(w_adder.rce_scaled_usage*(1-marg))
    w_adder['model_rate_adder'] = w_adder.marg_based_rate - w_adder.model_rate
    return w_adder

def rate_adder_hist(df,marg):
    # creating a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=(10,8))
    
    # assigning a graph to each ax
    sns.boxplot(data=df, x="model_rate_adder", ax=ax_box, color='c',boxprops=dict(alpha=1, label=f'std dev: {round(df.model_rate_adder.std(),3)}'))
    sns.histplot(data=df, x="model_rate_adder", ax=ax_hist, color='c').set_title(f"Model rate adders for a {int(100*marg)}% profit margin")
    mean_adder = df.model_rate_adder.mean()
    plt.axvline(x=mean_adder, label=f'avg rate adder: ${ round(mean_adder,3)}', color='m', linestyle='--')

    # Remove x axis name for the boxplot
    ax_box.set(xlabel='')
    ax_box.legend()
    ax_hist.set_xlabel("model rate adder ($)",fontsize=20)
    ax_hist.set_ylabel("Count",fontsize=20)
    plt.legend()
    st.pyplot(f)

def rate_adder_terms(df):
    fig = plt.figure(figsize=(10, 8))
    # sns.countplot(x = "year", data = df)
    box_plot = sns.boxplot(data=df, x="term", y="model_rate_adder",palette=['r','m','y','g','c'])
    box_plot.set_title("Model rate adders by term")
    medians = df.groupby('term')['model_rate_adder'].median().values
    medians = [round(i,3) for i in medians]
    vertical_offset = df['model_rate_adder'].median() * 0.2 # offset from median for display
    for xtick in box_plot.get_xticks():
        box_plot.text(xtick, medians[xtick] + vertical_offset, medians[xtick], 
                      horizontalalignment='center',size='x-large',color='w',weight='semibold')
    st.pyplot(fig)

def create_dol_per_rce_df(df):
    dol_per_rce = df.groupby('utility_account_number',as_index=False, sort=False).agg(acct_pred_tot_charge = ("model_tot_charge","sum"),
                                                acct_actual_tot_charge = ("computed_tot_charge","sum"),
                                                actual_rce = ("rce","mean"),
                                                pred_rce = ("pred_rce","mean"))
    dol_per_rce['dol_per_rce_model_actual'] = dol_per_rce['acct_actual_tot_charge']/dol_per_rce['actual_rce']
    dol_per_rce['dol_per_rce_model_pred'] = dol_per_rce['acct_pred_tot_charge']/dol_per_rce['pred_rce'] 
    return dol_per_rce

def dpr_hists(df):
    fig = plt.figure(figsize=(10,8))
    if len(df) > 5000:
        num_bins = 150
    elif 1000 < len(df) < 5000:
        num_bins = 50
    else: 
        num_bins = 30
    sns.histplot(df['dol_per_rce_model_actual'], bins=num_bins, color='m', alpha=0.75, label='model "actual" $/rce')
    sns.histplot(df['dol_per_rce_model_pred'], bins=num_bins, color='c', alpha=0.75, label="model pred $/rce")
    plt.axvline(x=df['dol_per_rce_model_actual'].median(), label=f'median "actual" \$/rce: ${ round(df["dol_per_rce_model_actual"].median(),2)}', color='m', linestyle='--')
    plt.axvline(x=df['dol_per_rce_model_pred'].median(), label=f'median pred \$/rce: ${ round(df["dol_per_rce_model_pred"].median(),2)}', color='c', linestyle='--')
    #plt.ylabel("Count")
    plt.xlabel("dollars/rce")
    #plt.xlim([0,5000])
    plt.title("\$/rce model actual vs \$/rce model prediction")
    plt.legend()
    st.pyplot(fig)


def dol_per_rce_diff(df):
    fig = plt.figure(figsize=(10,8))
    dollar_per_rce_diff= df.dol_per_rce_model_actual - df.dol_per_rce_model_pred
    sns.histplot(dollar_per_rce_diff,color='grey',bins=50)
    mean_diff = dollar_per_rce_diff.mean()
    median_diff = dollar_per_rce_diff.median()
    if mean_diff >= 0:
        plt.axvline(x=mean_diff, label=f'avg diff btw "actual" and pred: ${ round(mean_diff,2)}', color='c', linestyle='-')
        plt.axvline(x=median_diff, label=f'median diff btw "actual" and pred: ${ round(median_diff,2)}', color='k', linestyle='--')
    else:
        plt.axvline(x=mean_diff, label=f'avg diff btw "actual" and pred: - ${ np.abs(round(mean_diff,2))}', color='c', linestyle='-')
        plt.axvline(x=median_diff, label=f'median diff btw "actual" and pred: - ${ np.abs(round(median_diff,2))}', color='k', linestyle='--')
    plt.xlabel("\$/rce model actual - \$/rce model predictions")
    #plt.ylabel("Count")
    plt.title("Difference between \$/rce model actual and \$/rce model predictions")
    plt.legend()
    st.pyplot(fig)

def create_monthly_dpr_df(df:pd.DataFrame):
    v1 = df.groupby(['month','utility_account_number'], as_index=False)[['computed_tot_charge','rce','model_tot_charge','pred_rce']].sum()
    v2 = v1.groupby('month', as_index=False).agg(computed_rev = ("computed_tot_charge","sum"),
                                                tot_real_rce = ("rce","sum"),
                                                model_rev = ("model_tot_charge","sum"),
                                                tot_pred_rce = ("pred_rce","sum"))
    v2['computed_dol_per_rce'] = v2.computed_rev/v2.tot_real_rce
    v2['pred_dol_per_rce'] = v2.model_rev/v2.tot_pred_rce
    return v2

def monthly_dpr(df: pd.DataFrame):
    fig = plt.figure(figsize=(20,20))
    yticklabels = df['month']
    yticks = np.arange(0, len(df), 1)
    # The horizontal plot is made using the hline function
    mins = df[['computed_dol_per_rce','pred_dol_per_rce']].min(axis=1)
    maxes = df[['computed_dol_per_rce','pred_dol_per_rce']].max(axis=1)
    plt.hlines(y=yticks, xmin=mins, xmax=maxes, color='grey', alpha=0.75)
    # tuple of (x,y) coords
    for i,pair in enumerate(tuple(zip(mins +(maxes-mins-2)/2,yticks))):
        plt.annotate(round((maxes-mins)[i],3),pair, fontsize=15)
    plt.scatter(df['computed_dol_per_rce'], yticks, s=80, color='cyan', alpha=1, label='model "actual" \$/rce')
    plt.scatter(df['pred_dol_per_rce'], yticks, s=80, color='magenta', alpha=1 , label='model pred \$/rce')
    plt.legend()
    
    # Add title and axis names
    #plt.yticks(yticks,df['month'].apply(lambda x: x.strftime('%b %Y')))
    # use below if loaded in data
    plt.yticks(yticks,df['month'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d').strftime('%b %Y')))
    plt.title(f"Comparison of the model actual \$/rce and the model pred \$/rce each month", loc='center')
    plt.xlabel('dollars per rce')
    plt.ylabel('month')
    # Show the graph
    st.pyplot(fig)

if option == 'Rieman Backtest':
    st.header('Here we\'ll backtest the rieman model for contracts between a range of selected dates.')
    st.write('The backtest shows us the "actual" and predicted value of contracts signed between your range of selected dates. \
        \n This dashboard does the following: \n 1. Load in your csv containing the "actual" and predicted revenue based on the actual \
        \n 2. Select profit margin and visualize the rate adder needed to produce that margin \
        \n 3. Visualize the $/rce produced by the model vs the "actual"')

    # sdate,edate,_ = st.columns([1,1,3])
    # with sdate:
    #     start_date = st.date_input('start date') #this is a datetime dtype
    #     #st.write(start_date)
    # with edate:
    #     end_date = st.date_input('end date') #this is a datetime dtype
    #     #st.write(end_date) 

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        chunk = pd.read_csv(uploaded_file,dtype={'utility_account_number':str},chunksize=100000,low_memory=False)
        with_margs = pd.concat(chunk)
        st.write(with_margs)

        st.write("-"*20)
# -------------- MODEL COMPUTED METRICS ---------------------#
        computed_avg_rate, computed_gross_cost, computed_revenue, computed_margin = st.columns(4)
        with computed_avg_rate:
            comp_avg_rate = with_margs.computed_rate.mean()
            model_pred_avg_rate = with_margs.model_rate.mean()
            st.metric(label='Computed average rate', 
                value=f"${millify(comp_avg_rate, precision=3)}",
                delta=millify(comp_avg_rate - model_pred_avg_rate,precision=3) 
                )
        with computed_gross_cost:
            comp_tot_cost = with_margs.computed_gross_cost.sum()
            model_pred_cost = with_margs.pred_gross_cost.sum()
            st.metric(label='Computed total gross cost', 
                value=f"${millify(comp_tot_cost, precision=2)}",
                delta=millify(comp_tot_cost - model_pred_cost,precision=2)
                )
        with computed_revenue:
            comp_rev = with_margs.computed_tot_charge.sum()
            model_rev = with_margs.model_tot_charge.sum()
            st.metric(label='Computed total revenue', 
                value=f"${millify(comp_rev, precision=2)}",
                delta=millify(comp_rev - model_rev,precision=2)
                )
        with computed_margin:
            comp_marg = with_margs.computed_margin.sum()
            model_marg = with_margs.model_margin.sum()
            st.metric(label='Computed margin', 
                value=f"${millify(comp_marg, precision=2)}",
                delta=millify(comp_marg - model_marg,precision=2)
                )

    # -------------- MODEL PREDICTIONS METRICS ---------------------#
        model_avg_rate, model_gross_cost, model_revenue, model_margin = st.columns(4)
        with model_avg_rate:
            model_pred_avg_rate = with_margs.model_rate.mean()
            st.metric(label='Model average rate', 
                value=f"${millify(model_pred_avg_rate, precision=3)}"
                )
        with model_gross_cost:
            model_pred_cost = with_margs.pred_gross_cost.sum()
            st.metric(label='Model total gross cost', 
                value=f"${millify(model_pred_cost, precision=2)}"
                )
        with model_revenue:
            model_rev = with_margs.model_tot_charge.sum()
            st.metric(label='Model total revenue', 
                value=f"${millify(model_rev, precision=2)}"
                )
        with model_margin:
            model_marg = with_margs.model_margin.sum()
            st.metric(label='Model margin', 
                value=f"${millify(model_marg, precision=2)}"
                )

        st.write("-"*20)

        pm, pm_text = st.columns(2)
        with pm:
            profit_margin = st.number_input(label="Target profit margin as a percentage", min_value=0, max_value=100)
            set_marg = profit_margin/100
            w_adder = calc_rate_adder(with_margs,set_marg)
        with pm_text:
            st.write(f"\n Compute the rate adder to the modeled rates to get the desired profit margin of %{profit_margin}")
            st.write(w_adder.head())

        st.write("-"*20)
        _,head,_ = st.columns(3)
        with head:
            st.header("Visualizing the rate adder")

        adder_hist, adder_by_term = st.columns(2)
        with adder_hist:
            rate_adder_hist(w_adder,set_marg)

        with adder_by_term:
            rate_adder_terms(w_adder)

        st.write("-"*20)
        _,head,_ = st.columns(3)
        with head:
            st.header("Visualizing the dollars/rce")

        dol_per_rce = create_dol_per_rce_df(with_margs)

        dol_per_rce_hists, dol_per_rce_diffs = st.columns(2)
        with dol_per_rce_hists:
            dpr_hists(dol_per_rce)

        with dol_per_rce_diffs:
            dol_per_rce_diff(dol_per_rce)
        
        v2 = create_monthly_dpr_df(with_margs) 
        monthly_dpr(v2)



################## CODE FOR BACKTEST RETENTION BELOW #####################
# Retention Prediction Backtest for Resi and Comm accounts between .5 and 10 RCEs signed between Jan 2021 and Mar 2021
# Purpose: Show that Contract GROOVE model allows us to hedge less energy to cover true expected load of contracts

# 1. Find all applicable contracts signed in commercial_base
# 2. Request historical book_forecast_detail report for Jan 2021
# 3. Filter to just accounts in step 1, and filter to just contract bills
# 4. Prepare report of same accounts on book_backcast_detail or ltv.present_value
# 5. Left join backcast data to forecast data, replacing forecast usage with actual usage when able
# 6. Format the book forecast data into contract groove format
# 7. Send forecast thru groove, predicting retention on all forecast bills
# 8. Compute cummulative drop per bill
# 9. Rollup into expected usage by state and month
# - usage * cummulative retainment
# 10. take historical data, rollup actual usage by state and month
# 11. plot monthly diff between historical and actual usage by state and month
# - line plot

if option == "Backtest Retention":
    st.header('hi there, here we\'ll backtest the modeled retention for contracts between a range of selected dates.')
    st.write('The backtest shows us what the model would have predicted for retention on contracts signed between your range of selected dates. Then compares the predicted retention to the actual retention.')
    # uploaded_file = st.file_uploader("Choose a file")
    # if uploaded_file is not None:
    #     chunk = pd.read_csv(uploaded_file,dtype={'utility_account_number':str},chunksize=100000,low_memory=False)
    #     df = pd.concat(chunk)
    #     st.write(df)

    file,sdate,edate = st.columns([3,1,1])
    with file:
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            chunk = pd.read_csv(uploaded_file,dtype={'utility_account_number':str},chunksize=100000,low_memory=False)
            df = pd.concat(chunk)
            st.write(df)
    with sdate:
        start_date = st.date_input('start date') #this is a datetime dtype
        st.write(start_date)
    with edate:
        end_date = st.date_input('end date') #this is a datetime dtype
        st.write(end_date) #

    # to_plot = df[df.service_end_date < start_date.strftime('%Y-%m-%d')]
    # st.write(to_plot.head())
    # fig = px.line(to_plot, x=to_plot['service_end_date'].apply(lambda x: x[5:]), y=to_plot['cummulative_retention']*to_plot['use_adj'],color='org_type')
    # st.plotly_chart(fig, use_container_width=True)

################################################################################
# if option == 'LTV':
#     ltv = load_ltv()[0]
#     no_nulls_ltv = load_ltv()[1]
    
#     years=['2013-01-01','2014-01-01','2015-01-01','2016-01-01','2017-01-01','2018-01-01','2019-01-01','2020-01-01','2021-01-01']
#     year_pick = st.select_slider('Pick date to filter on. Only data after this date will be included in the below analysis:', options = years)#, format_func= lambda x: x.strftime("%m/%d/%Y"))
#     st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
#     st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-right:10px;padding-left:4px;}</style>', unsafe_allow_html=True)
#     target_metric = st.radio("Select a target metric to visualize:", ['margin','cogs','revenue','cost','usage','margin_percent'])
#     c = st.container()
#     group1 = st.radio("Select category to group your target metric on:", ['none','service_zone','sales_channel','term','service_start_date','rate_type','iso_zone','state_name'])
#     group2 = st.radio("Select a second category if you'd like:", ['none','service_zone','sales_channel','term','service_start_date','rate_type','iso_zone','state_name'])
#     tmp_df = metric_bar_plot_grouped_by(group1,group2,target_metric,no_nulls_ltv,year_pick)
    
    # st.write('## Let\'s use historical data to describe our least and most profitable customers by selected metrics')
    # filter_choice = st.radio('First, what should we use to filter outliers in the data?',['interquartile range','standard deviation'])
    # if filter_choice == 'interquartile range':
    #     filtered_no_nulls_ltv = filterby_iqr(no_nulls_ltv)
    # else:
    #     pass
    # text,sel = st.columns([5,1])
    # with text:
    #     #this is not yet true & code needs to be modified to take the tmp df created by the user
    #     st.write("#### Based on the constraints you\'ve placed on the data these categories describe the highest and lowest:")
    # #with sel:
    #     col_choice = st.selectbox("", ('margin','cogs','revenue','margin_percent'))
    #     #st.write(col_choice[0])
    # labeled_df = create_quartiles(filtered_no_nulls_ltv,col_choice)
    # #st.write(labeled_df.head())
    # cat_df = just_categoricals(labeled_df)
    # #st.write(cat_df[cat_df[col_choice+'_labels'] == 'high '+str(col_choice)].service_zone.mode()[0])
    # sz,sales_chan,term,rate,iz,lp =  st.columns([1,1.5,0.5,1,1.5,0.5])
    # with sz:
    #     st.metric('service zone',cat_df[cat_df[col_choice+'_labels'] == 'high '+str(col_choice)].service_zone.mode()[0])
    # with sales_chan:
    #     st.metric('sales channel',cat_df[cat_df[col_choice+'_labels'] == 'high '+str(col_choice)].sales_channel.mode()[0])
    # with term:
    #     st.metric('service zone',cat_df[cat_df[col_choice+'_labels'] == 'high '+str(col_choice)].term.mode()[0])
    # with rate:
    #     st.metric('rate type',cat_df[cat_df[col_choice+'_labels'] == 'high '+str(col_choice)].rate_type.mode()[0])
    # with iz:
    #     st.metric('iso zone',cat_df[cat_df[col_choice+'_labels'] == 'high '+str(col_choice)].iso_zone.mode()[0])
    # with lp:
    #     st.metric('load profile',cat_df[cat_df[col_choice+'_labels'] == 'high '+str(col_choice)].load_profile.mode()[0]

################################################################################
# survival plot function move to top after editing
def other_by_st_and_org_plot_kmf(df:pd.DataFrame, state:str):
    orgs = sorted(list(df.org_type.unique()))
    colors=cm.rainbow(np.linspace(0,1,len(orgs)))
    #master_table = []
    fig, ax = plt.subplots(figsize=(14, 8))
    for i,org in enumerate(orgs):
        new = df[(df.org_type == org) & (df.state_name == state)]
        #st.write(new.head())
        if new.shape[0] == 0:
            print(f'empty df for {org}')
            continue
        else:
            kmf = KaplanMeierFitter()
            ax = kmf.fit_right_censoring(new['dur_in_mos'], new['dropped'],label=f"{(state).upper()}, {org}").plot_survival_function(color=colors[len(orgs)-1-i])
            ax.legend(fontsize=16)
            ax.set_xlabel('months',fontsize=18)
            ax.set_ylabel('survival probability',fontsize=18)
    st.pyplot(fig)

def by_term(df:pd.DataFrame):
    terms = [1,3,6,12,24,36]
    colors=cm.rainbow(np.linspace(0,1,len(terms)))
    fig,ax = plt.subplots(figsize=(14, 8))
    for i,t in enumerate(terms):
        new = df[df.term == t]
        #st.write(new.head())
        if new.shape[0] == 0:
            print(f'empty df for {t}')
            continue
        else:
            kmf = KaplanMeierFitter()
            ax = kmf.fit_right_censoring(new['dur_in_mos'], new['dropped'],label=f"term: {t} mos.").plot_survival_function(color=colors[len(terms)-1-i])#,ax=ax)
            #add_at_risk_counts(kmf, fontsize=20, rows_to_show=['At risk'],ax=fig) #, ax=ax,fig=fig)#, ax=ax,fig=fig)
            ax.legend(fontsize=18)
            ax.set_xlabel('dur_in_mos',fontsize=18)
            #fig.set_xticks(np.arange(0, df.month.max()+1, step=5))
            ax.set_ylabel('survival probability',fontsize=18)
    st.pyplot(fig)

# loading in and cleaning data move elsewhere once cleaned up
# @st.cache(allow_output_mutation=True)
# def load_clean_drop_data():
#     #this hardcodes in using the groove model step 1 output
#     df = pd.read_csv('/Users/tmarianchuk/Desktop/variable_rate_setting_model/groove-model/Preprocessing/allout_2021-10-26.csv',dtype={'utility_account_number':'str','load_profile':'str'})
#     #below is cleaning from groove step 2 preprocessing
#     #df['m2'] = pd.to_datetime(df.midpoint).dt.date.apply(lambda x: datetime.date(x.year,x.month,1))
#     #only care about pam, filter nulls out
#     df = df[df.org_type.isin(['PAM','Inside Sales','Vendor','Online'])][~df.use_adj.isnull()][~df.last_bill.isnull()][~df.zipcode.isnull()][~df.contract_rate.isnull()][~df.longitude.isnull()]

#     #take rows up to two months from today
#     #tdate = datetime.date.today() - relativedelta(months=1)
#     #tdate = datetime.date(tdate.year,tdate.month,1)

#     #print('using bills with usage up to:',tdate)

#     #filter outlier customers
#     dat = df.copy()
#     #dat = dat[pd.to_datetime(dat.m2).dt.date <= tdate]
#     #del dat['m2']
#     dat = dat[dat.last_rate.between(0,0.3)]
#     dat = dat[dat.last_bill.between(0,200)]
#     dat = dat[dat.last_use.between(0,2000)]
#     dat = dat[dat.bill_adj.between(0,200)]
#     dat = dat[dat.use_adj.between(0,2000)]
#     dat = dat[dat.rate.between(0,0.3)]
#     dat = dat[dat.rce.between(0,5)]
#     dat = dat[dat.vdays.between(-45,9999999)]
#     dat = dat[dat.period.between(14,60)]

#     print('after filtering outliers the data has shape: {}'.format(dat.shape))

#     # making the df compatible with survival analysis
#     # distinct on to remove duplicate months
#     dat.drop_duplicates(subset = ['utility_account_number',
#                             'service_zone','request_id',
#                             'service_start_date'], inplace=True)
#     # then groupby service_zone, utility_account_number and request_id and do a cumulative account of the rows which are 
#     # luckily already by month
#     dat['dur_in_mos'] = dat.groupby(['service_zone',
#                                                             'utility_account_number',
#                                                             'request_id']).cumcount() 
#     dat.drop_duplicates(subset=['utility_account_number','enroll_on'],keep='last',inplace=True)                                                                                       
#     return(dat)

#drop_df = load_clean_drop_data() 
#SO much faster displaying when you cache the data load and clean process in a func

####################################################################################

################## CODE PERTAINING TO SCRATCH PAGE LIES HERE #####################
if option == 'scratch':
    st.subheader('Here\'s your scratch page to trial things on')

    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/violin_data.csv")

    fig = go.Figure()

    days_yes = df['day'][ df['smoker'] == 'Yes']
    grouped_labels_yes = days_yes.map({"Thur":"A", "Fri":"A", "Sat":"B", "Sun":"B"})
    days_no = df['day'][ df['smoker'] == 'No']
    grouped_labels_no = days_no.map({"Thur":"A", "Fri":"A", "Sat":"B", "Sun":"B"})

    fig.add_trace(go.Violin(x=[grouped_labels_yes, days_yes],
                            y=df['total_bill'][ df['smoker'] == 'Yes' ],
                            legendgroup='Yes', scalegroup='Yes', name='Yes',
                            side='negative',
                            line_color='cyan')
                )
    fig.add_trace(go.Violin(x=[grouped_labels_no, days_no],
                            y=df['total_bill'][ df['smoker'] == 'No' ],
                            legendgroup='No', scalegroup='No', name='No',
                            side='positive',
                            line_color='magenta')
                )
    fig.update_traces(meanline_visible=True)
    fig.update_layout(violingap=0, violinmode='overlay',
                  title={'text':'Total bill by day for smokers vs non-smokers'},
                    # autosize=False,
                    # width=800,
                    # height=500,
                    # margin=dict(
                    #         l=50,
                    #         r=300,
                    #         b=100,
                    #         t=100,
                    #         pad=10),
                    #paper_bgcolor="white",#"LightSteelBlue",
                    )
    fig.update_yaxes(tickprefix="$") 
    fig.update_xaxes(autorange=True, dividerwidth=0.8) 
    st.plotly_chart(fig, use_container_width=True)

    st.write(df.head(20))

    # below is code for creating horizontal radio buttons
    # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
    # st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-right:6px;padding-left:4px;}</style>', unsafe_allow_html=True)
    # genre = st.radio("What's your favorite movie genre?",
    #             ('Comedy', 'Drama', 'Documentary'))

    ####################################################################################
