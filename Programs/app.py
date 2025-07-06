import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="COVID-19 EDA Dashboard")

# --- Data Loading ---
@st.cache_data
def load_data():
    main_df = pd.read_csv('../Data/covid_19_data.csv')
    geo_df = pd.read_csv('../Data/time_series_covid_19_confirmed.csv')
    geo_df['Country/Region'] = geo_df['Country/Region'].replace({
        'Mainland China': 'China',
        'US': 'United States',
        'Korea, South': 'South Korea',
        'Taiwan': 'Taiwan',
        'Iran (Islamic Republic of)': 'Iran',
        'Russian Federation': 'Russia',
        'Czechia': 'Czech Republic',
        'Vietnam': 'Viet Nam'
    })
    geo_by_country = geo_df.groupby('Country/Region')[['Lat', 'Long']].mean().reset_index()
    main_df['Country/Region'] = main_df['Country/Region'].replace({
        'Mainland China': 'China',
        'US': 'United States',
        'Korea, South': 'South Korea',
        'Taiwan': 'Taiwan',
        'Iran (Islamic Republic of)': 'Iran',
        'Russian Federation': 'Russia',
        'Czechia': 'Czech Republic',
        'Vietnam': 'Viet Nam'
    })
    main_df = main_df.merge(geo_by_country, on='Country/Region', how='left')
    main_df = main_df.dropna(subset=['Lat'])
    main_df['ObservationDate'] = pd.to_datetime(main_df['ObservationDate'])
    main_df['Month'] = main_df['ObservationDate'].dt.month
    main_df = main_df.sort_values(['Province/State', 'Country/Region', 'ObservationDate'], ascending=[False, False, True])
    main_df['key'] = main_df['Country/Region'].astype(str) + ' - ' + main_df['Province/State'].astype(str)
    def safe_diff(group, col):
        if len(group) == 1:
            return group[col]
        else:
            return group[col].diff().fillna(group[col])
    main_df['Confirmed_diff'] = main_df.groupby('key', group_keys=False).apply(lambda g: safe_diff(g, 'Confirmed'))
    main_df['Deaths_diff'] = main_df.groupby('key', group_keys=False).apply(lambda g: safe_diff(g, 'Deaths'))
    if 'Recovered' in main_df.columns:
        main_df['Recovered_diff'] = main_df.groupby('key', group_keys=False).apply(lambda g: safe_diff(g, 'Recovered'))
    main_df = main_df[
        (main_df['Confirmed_diff'] >= 0) &
        (main_df['Deaths_diff'] >= 0) &
        (main_df['Recovered_diff'] >= 0)
    ]
    return main_df

main_df = load_data()

# --- EDA Metrics ---
st.title("COVID-19 EDA Dashboard")

main_df['Death_Rate_orig'] = np.where(
    main_df['Confirmed'] > 0,
    (main_df['Deaths'] / main_df['Confirmed']) * 100,
    np.nan
)
main_df['Recovery_Rate_orig'] = np.where(
    main_df['Confirmed'] > 0,
    (main_df['Recovered'] / main_df['Confirmed']) * 100,
    np.nan
)

total_confirmed = main_df.groupby('Country/Region')['Confirmed'].max().sum()
total_deaths = main_df.groupby('Country/Region')['Deaths'].max().sum()
total_recovered = main_df.groupby('Country/Region')['Recovered'].max().sum()
recovery_rate = (total_recovered / total_confirmed * 100) if total_confirmed > 0 else None
death_rate = (total_deaths / total_confirmed * 100) if total_confirmed > 0 else None
num_countries = main_df['Country/Region'].nunique()
num_provinces = main_df['Province/State'].nunique()
date_min = main_df['ObservationDate'].min()
date_max = main_df['ObservationDate'].max()

st.header("Overall Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Confirmed", f"{int(total_confirmed):,}")
col2.metric("Total Deaths", f"{int(total_deaths):,}")
col3.metric("Total Recovered", f"{int(total_recovered):,}")
col4.metric("Countries/Regions", f"{num_countries}")

col5, col6, col7 = st.columns(3)
col5.metric("Recovery Rate", f"{recovery_rate:.2f}%" if recovery_rate is not None else "N/A")
col6.metric("Death Rate", f"{death_rate:.2f}%" if death_rate is not None else "N/A")
col7.metric("Provinces/States", f"{num_provinces}")

st.caption(f"Date Range: {date_min.date()} to {date_max.date()}")

# --- Daily Trends Plot ---
st.subheader("Daily New Trends: Confirmed, Recovered & Deaths")
daily_diff = main_df.groupby('ObservationDate')[['Confirmed_diff', 'Deaths_diff', 'Recovered_diff']].sum()
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
fig.add_trace(go.Scatter(x=daily_diff.index, y=daily_diff['Confirmed_diff'], name='Confirmed', line=dict(color='blue', width=2)), row=1, col=1)
if 'Recovered_diff' in daily_diff.columns:
    fig.add_trace(go.Scatter(x=daily_diff.index, y=daily_diff['Recovered_diff'], name='Recovered', line=dict(color='green', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=daily_diff.index, y=daily_diff['Deaths_diff'], name='Deaths', line=dict(color='red', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=daily_diff.index, y=daily_diff['Deaths_diff'], name='Deaths', line=dict(color='red', width=2, dash='dot')), row=2, col=1)
fig.update_layout(height=600, showlegend=True, legend=dict(x=0.01, y=1), plot_bgcolor='white', paper_bgcolor='white')
fig.update_xaxes(title_text="Date", row=2, col=1)
fig.update_yaxes(title_text="Number of Cases", row=1, col=1)
fig.update_yaxes(title_text="Number of Deaths", row=2, col=1)
st.plotly_chart(fig, use_container_width=True)

# --- Top 10 Countries Bar Chart ---
st.subheader("Top 10 Countries by Confirmed/Deaths/Recovered")
country_stats = main_df.groupby('Country/Region').agg({
    'Confirmed_diff': 'sum',
    'Deaths_diff': 'sum',
    'Recovered_diff': 'sum' if 'Recovered_diff' in main_df.columns else 'min'
}).reset_index()
metrics = {
    'Confirmed': {'col': 'Confirmed_diff', 'color': 'royalblue'},
    'Deaths': {'col': 'Deaths_diff', 'color': 'crimson'}
}
if 'Recovered_diff' in country_stats.columns:
    metrics['Recovered'] = {'col': 'Recovered_diff', 'color': 'seagreen'}

metric_choice = st.radio("Select metric", list(metrics.keys()), horizontal=True)
sorted_df = country_stats.sort_values(metrics[metric_choice]['col'], ascending=True).tail(10)
fig = go.Figure(go.Bar(
    name=metric_choice,
    y=sorted_df['Country/Region'],
    x=sorted_df[metrics[metric_choice]['col']],
    marker_color=metrics[metric_choice]['color'],
    orientation='h',
    text=sorted_df[metrics[metric_choice]['col']],
    textposition='inside'
))
fig.update_layout(title=f"Top 10 Countries by {metric_choice}", yaxis_title='Country/Region', xaxis_title='Count', height=500)
st.plotly_chart(fig, use_container_width=True)

# --- Recovery Rate Over Time ---
st.subheader("Mean Monthly Recovery Rate for Top 5 Countries (including USA)")
main_df['YearMonth'] = main_df['ObservationDate'].dt.to_period('M')
main_df['Recovery_Rate_orig'] = np.where(
    (main_df['Confirmed'] > 0) & (main_df['Recovered'].notna()),
    (main_df['Recovered'] / main_df['Confirmed']) * 100,
    np.nan
)
top5_countries = (
    main_df.groupby('Country/Region')['Confirmed_diff']
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .index
    .tolist()
)
if 'United States' not in top5_countries and 'United States' in main_df['Country/Region'].unique():
    top5_countries.append('United States')
top5_countries = list(dict.fromkeys(top5_countries))
top5_df = main_df[main_df['Country/Region'].isin(top5_countries)]
monthly_recovery = (
    top5_df.groupby(['Country/Region', 'YearMonth'])['Recovery_Rate_orig']
    .mean()
    .reset_index()
)
fig = go.Figure()
for country in top5_countries:
    country_data = monthly_recovery[monthly_recovery['Country/Region'] == country]
    fig.add_trace(go.Scatter(
        x=country_data['YearMonth'].astype(str),
        y=country_data['Recovery_Rate_orig'],
        mode='lines+markers',
        name=f"{country} (mean)"
    ))
fig.update_layout(
    title='Mean Monthly Recovery Rate (Original) for Top 5 Countries by Confirmed Cases (including USA)',
    xaxis_title='Month',
    yaxis_title='Mean Recovery Rate (%)',
    legend_title='Country',
    template='plotly_white',
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# --- Death Rate Over Time ---
st.subheader("Mean Monthly Death Rate for Top 5 Countries (including USA)")
main_df['Death_Rate_orig'] = np.where(
    (main_df['Confirmed'] > 0) & (main_df['Deaths'].notna()),
    (main_df['Deaths'] / main_df['Confirmed']) * 100,
    np.nan
)
monthly_death = (
    top5_df.groupby(['Country/Region', 'YearMonth'])['Death_Rate_orig']
    .mean()
    .reset_index()
)
fig = go.Figure()
for country in top5_countries:
    country_data = monthly_death[monthly_death['Country/Region'] == country]
    fig.add_trace(go.Scatter(
        x=country_data['YearMonth'].astype(str),
        y=country_data['Death_Rate_orig'],
        mode='lines+markers',
        name=f"{country} (mean)"
    ))
fig.update_layout(
    title='Mean Monthly Death Rate (Original) for Top 5 Countries by Confirmed Cases (including USA)',
    xaxis_title='Month',
    yaxis_title='Mean Death Rate (%)',
    legend_title='Country',
    template='plotly_white',
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# --- Interactive Map Visualization (Plotly scatter_geo) ---
st.subheader("World COVID-19 Map (by day and metric)")

main_df['ObservationDate'] = pd.to_datetime(main_df['ObservationDate'])
date_options = main_df['ObservationDate'].dt.date.unique()
date_options = sorted(date_options)

selected_date = st.selectbox("Select Date", date_options, index=len(date_options)-1)
metric = st.radio("Select Map Metric", ['Confirmed', 'Deaths', 'Recovered'], horizontal=True)

metric_col = {
    'Confirmed': 'Confirmed_diff',
    'Deaths': 'Deaths_diff',
    'Recovered': 'Recovered_diff'
}[metric]

filtered = main_df[main_df['ObservationDate'].dt.date == pd.to_datetime(selected_date)]
country_map_df = filtered.groupby('Country/Region').agg({
    'Confirmed_diff': 'sum',
    'Deaths_diff': 'sum',
    'Recovered_diff': 'sum' if 'Recovered_diff' in main_df.columns else 'min',
    'Lat': 'mean',
    'Long': 'mean'
}).reset_index()

country_map_df = country_map_df[country_map_df[metric_col] > 0]

fig = px.scatter_geo(
    country_map_df,
    lat='Lat',
    lon='Long',
    hover_name='Country/Region',
    size=metric_col,
    color=metric_col,
    projection='natural earth',
    title=f"{metric} Cases on {selected_date}",
    color_continuous_scale='OrRd' if metric == 'Confirmed' else ('Reds' if metric == 'Deaths' else 'Greens'),
    size_max=40,
)

fig.update_layout(
    geo=dict(bgcolor= 'rgba(0,0,0,0)'),
    margin={"r":0,"t":40,"l":0,"b":0}
)

st.plotly_chart(fig, use_container_width=True)

st.caption("Data source: Johns Hopkins University COVID-19 dataset")