
#Import libraries
import streamlit as st
import pandas as pd; import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots as ps
import plotly.express as px

# Page configuration
st.set_page_config(page_title="COVID-19 Data Visualization Dashboard",page_icon="🦠",layout="wide",initial_sidebar_state="expanded")

# Custom CSS for dark theme consistency
st.markdown("""
<style>
    .main {
        background-color: rgb(30, 30, 30);
    }
    .stSelectbox > div > div {
        background-color: rgb(50, 50, 50);
        color: white;
    }
    .stNumberInput > div > div > input {
        background-color: rgb(50, 50, 50);
        color: white;
    }
    .metric-container {
        background-color: rgb(50, 50, 50);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-title {
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .metric-value {
        color: #1f77b4;
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the processed COVID-19 data from notebook output"""
    try:
        datalong = pd.read_csv('../Output/datalong.csv')
        total_per_country_wide = pd.read_csv('../Output/total_per_country_wide.csv')
        datalong['Date'] = pd.to_datetime(datalong['Date'])
        
        if 'Daily_cases' not in datalong.columns:
            datalong['Daily_cases'] = datalong.groupby(['Country/Province', 'Status'])['Cases'].diff().fillna(0)
            datalong.loc[datalong['Daily_cases'] < 0, 'Daily_cases'] = 0
        return datalong, total_per_country_wide
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def get_country_list(datalong):
    """Get list of all countries for search functionality"""
    countries = list(datalong['Country/Province'].unique())
    countries = sorted(countries)
    countries.insert(0, 'Global')
    return countries

def calculate_metrics(datalong, total_per_country_wide, country_choice):
    """Calculate metrics for selected country or global using notebook logic"""
    total_per_country = total_per_country_wide.melt(id_vars=['Country/Province', 'Rank', 'average_daily_cases', 'average_daily_deaths'], 
        value_vars=['confirmed', 'death', 'recovered', 'active', 'death_rate', 'recovery_rate'],var_name='Status', value_name='Cases').dropna()
    
    if country_choice == 'Global':
        total = total_per_country.groupby('Status').agg({'Cases': 'sum', 'Country/Province': 'nunique'}).reset_index().copy()
        total.loc[total['Status'].isin(['recovery_rate', 'death_rate']),"Cases"] = total.loc[total['Status'].isin(['recovery_rate', 'death_rate']),"Cases"] / total.loc[total['Status'].isin(['recovery_rate', 'death_rate']),"Country/Province"]
        
        # Current week (last 7 days)
        d = datalong[datalong['Date'] > (datalong['Date'].max() - pd.Timedelta(days=6))].copy()
        d = d.groupby(['Status']).sum('Daily_cases').reset_index()
        # Previous week  
        week = 1
        dw = datalong[datalong['Date'].between(datalong['Date'].max() - pd.Timedelta(days=2*(6*week + 1)), datalong['Date'].max() - pd.Timedelta(days=(6*week + 1)))].copy()
        dw = dw.groupby(['Status']).sum('Daily_cases').reset_index()
        
        # Extract totals
        confirmed_total = total.loc[total['Status'] == 'confirmed', 'Cases'].sum()
        deaths_total = total.loc[total['Status'] == 'death', 'Cases'].sum()
        recovered_total = total.loc[total['Status'] == 'recovered', 'Cases'].sum()
        active_total = total.loc[total['Status'] == 'active', 'Cases'].sum()
        death_rate = total.loc[total['Status'] == 'death_rate', 'Cases'].mean() if not total.loc[total['Status'] == 'death_rate', 'Cases'].empty else 0
        recovery_rate = total.loc[total['Status'] == 'recovery_rate', 'Cases'].mean() if not total.loc[total['Status'] == 'recovery_rate', 'Cases'].empty else 0
        
        rank = 1
    else:
        total = total_per_country[total_per_country['Country/Province'].str.contains(country_choice, regex=True, na=False)].groupby('Status').agg({'Cases': 'sum', 'Country/Province': 'nunique'}).reset_index().copy()
        total.loc[total['Status'].isin(['recovery_rate', 'death_rate']),"Cases"] = total.loc[total['Status'].isin(['recovery_rate', 'death_rate']),"Cases"] / total.loc[total['Status'].isin(['recovery_rate', 'death_rate']),"Country/Province"]
        
        # Current week
        country = datalong[(datalong['Country/Province'].str.contains(country_choice, regex=True, na=False))].copy()
        if country.empty:
            return None
        d = country[country['Date'] > (country['Date'].max() - pd.Timedelta(days=6))].copy()
        d = d.groupby(['Status']).sum('Daily_cases').reset_index()
        # Previous week
        week = 1
        dw = country[country['Date'].between(country['Date'].max() - pd.Timedelta(days=2*(6*week + 1)), country['Date'].max() - pd.Timedelta(days=(6*week + 1)))].copy()
        dw = dw.groupby(['Status']).sum('Daily_cases').reset_index()
        
        # Extract totals
        confirmed_total = total.loc[total['Status'] == 'confirmed', 'Cases'].sum() if not total.loc[total['Status'] == 'confirmed', 'Cases'].empty else 0
        deaths_total = total.loc[total['Status'] == 'death', 'Cases'].sum() if not total.loc[total['Status'] == 'death', 'Cases'].empty else 0
        recovered_total = total.loc[total['Status'] == 'recovered', 'Cases'].sum() if not total.loc[total['Status'] == 'recovered', 'Cases'].empty else 0
        active_total = total.loc[total['Status'] == 'active', 'Cases'].sum() if not total.loc[total['Status'] == 'active', 'Cases'].empty else 0
        
        death_rate = total.loc[total['Status'] == 'death_rate', 'Cases'].mean() if not total.loc[total['Status'] == 'death_rate', 'Cases'].empty else 0
        recovery_rate = total.loc[total['Status'] == 'recovery_rate', 'Cases'].mean() if not total.loc[total['Status'] == 'recovery_rate', 'Cases'].empty else 0
        
        # Get rank from total_per_country_wide
        country_rank_data = total_per_country_wide[total_per_country_wide['Country/Province'].str.contains(country_choice, regex=True, na=False)]
        rank = country_rank_data['Rank'].min() if not country_rank_data.empty else 0
    
    # Extract recent data (current and previous week)
    current_confirmed = int(d.loc[d['Status'] == 'confirmed', 'Daily_cases'].values[0]) if not d.loc[d['Status'] == 'confirmed', 'Daily_cases'].empty else 0
    current_death = int(d.loc[d['Status'] == 'death', 'Daily_cases'].values[0]) if not d.loc[d['Status'] == 'death', 'Daily_cases'].empty else 0
    current_recovered = int(d.loc[d['Status'] == 'recovered', 'Daily_cases'].values[0]) if not d.loc[d['Status'] == 'recovered', 'Daily_cases'].empty else 0
    current_active = int(d.loc[d['Status'] == 'active', 'Daily_cases'].values[0]) if not d.loc[d['Status'] == 'active', 'Daily_cases'].empty else 0
    
    return {'confirmed_total': int(confirmed_total) if not pd.isna(confirmed_total) else 0,'deaths_total': int(deaths_total) if not pd.isna(deaths_total) else 0,
        'recovered_total': int(recovered_total) if not pd.isna(recovered_total) else 0,'active_total': int(active_total) if not pd.isna(active_total) else 0,
        'death_rate': death_rate if not pd.isna(death_rate) else 0,'recovery_rate': recovery_rate if not pd.isna(recovery_rate) else 0,
        'recent_confirmed': current_confirmed,'recent_deaths': current_death,'recent_recovered': current_recovered,'recent_active': current_active,
        'rank': int(rank) if not pd.isna(rank) else 0
    }

def create_time_series_plots(datalong, country_choice, total_per_country_wide):
    """Create time series plots for selected country"""
    if country_choice == 'Global':
        confirmed_data = datalong[datalong['Status'] == 'confirmed'].copy()
        death_data = datalong[datalong['Status'] == 'death'][['Country/Province', 'Date', 'Cases','Daily_cases']].copy()
        death_data.columns = ['Country/Province', 'Date', 'Death','Daily_death']
        d = total_per_country_wide
    else:
        confirmed_data = datalong[(datalong['Country/Province'].str.contains(country_choice, regex=True, na=False)) & (datalong['Status'] == 'confirmed')].copy()
        death_data = datalong[(datalong['Country/Province'].str.contains(country_choice, regex=True, na=False)) & (datalong['Status'] == 'death')][['Country/Province', 'Date', 'Cases','Daily_cases']].copy()
        death_data.columns = ['Country/Province', 'Date', 'Death','Daily_death']
        d = total_per_country_wide[total_per_country_wide['Country/Province'].str.contains(country_choice, regex=True, na=False)]
    global_daily = confirmed_data.groupby('Date').agg({'Cases': 'sum','Daily_cases': 'sum'}).reset_index()
    death_daily = death_data.groupby('Date').agg({'Death': 'sum','Daily_death': 'sum'}).reset_index()
    global_daily = global_daily.merge(death_daily, on=['Date'], how='left')
    
    # Create subplots
    fig = ps(rows=2, cols=2, 
             subplot_titles=[f'{country_choice} Cumulative Cases', f'{country_choice} Daily New Cases',
                           f'{country_choice} Cumulative Deaths', f'{country_choice} Daily New Deaths'],
             specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
             horizontal_spacing=0.08, vertical_spacing=0.1)
    
    # Plot 1: Global cumulative cases
    fig.add_trace(go.Scatter(x=global_daily['Date'], y=global_daily['Cases'],
        mode='lines',line=dict(color='#1f77b4', width=5),fill='tonexty',fillcolor='rgba(31,119,180,0.3)',name='',
        hovertemplate='Date: %{x}<br>Cases: %{y:,}<br>Deaths: %{customdata[0]:,}<extra></extra>',customdata=global_daily[['Death']].values
    ), row=1, col=1)
    # Plot 2: Global daily new cases
    fig.add_trace(go.Scatter(x=global_daily['Date'], y=global_daily['Daily_cases'],
        mode='lines',line=dict(color='#1f77b4', width=3),fill='tozeroy',fillcolor='rgba(31,119,180,0.3)', name='',
        hovertemplate='Date: %{x}<br>Cases: %{y:,}<br>Deaths: %{customdata[0]:,}<extra></extra>',customdata=global_daily[['Daily_death']].values
    ), row=1, col=2)
    # Plot 3: Global cumulative deaths
    fig.add_trace(go.Scatter(x=global_daily['Date'], y=global_daily['Death'],
        mode='lines',line=dict(color='#d62728', width=5),fill='tonexty',fillcolor='rgba(214,39,40,0.3)',name='',
        hovertemplate='Date: %{x}<br>Cases: %{y:,}<br>Confirmed: %{customdata[0]:,}<extra></extra>',customdata=global_daily[['Cases']].values
    ), row=2, col=1)
    # Plot 4: Global daily new deaths
    fig.add_trace(go.Scatter(x=global_daily['Date'], y=global_daily['Daily_death'],
        mode='lines',line=dict(color='#d62728', width=3),fill='tozeroy',fillcolor='rgba(214,39,40,0.3)', name='',
        hovertemplate='Date: %{x}<br>Cases: %{y:,}<br>Confirmed: %{customdata[0]:,}<extra></extra>',customdata=global_daily[['Daily_cases']].values
    ), row=2, col=2)
    
    # Update layout
    fig.update_layout(width=1600, height=1000,
        title={'text': f'📊 {country_choice}: COVID-19 New Cases & Deaths Over Time','x': 0.5,'xanchor': 'center','font': {'size': 20, 'color': 'white'}},
        showlegend=False, plot_bgcolor='rgb(30, 30, 30)',paper_bgcolor='rgb(30, 30, 30)', font=dict(color='white'))
    fig.update_xaxes(showgrid=False, tickfont=dict(size=10, color='white'))
    fig.update_yaxes(showgrid=True,gridcolor='rgb(100, 100, 100)',gridwidth=0.5,griddash='dash', tickfont=dict(size=15, color='white'))
    
    # Add annotations
    max_daily_case_date = global_daily.loc[global_daily['Daily_cases'].idxmax(), 'Date']
    max_daily_case = global_daily['Daily_cases'].max()
    max_daily_death_value = global_daily['Daily_death'].iloc[global_daily['Daily_cases'].idxmax()]
    fig.add_annotation(x=max_daily_case_date,y=max_daily_case,text=f"Peak Daily Cases<br>Date: {max_daily_case_date.strftime('%Y-%m-%d')}<br>Confirmed: {max_daily_case:,.0f}<br>Deaths: {max_daily_death_value:,.0f}",
    showarrow=False,arrowhead=1,arrowsize=1,arrowwidth=3,arrowcolor="white",ax=20,ay=-30,bgcolor="#1f77b4",bordercolor="white",borderwidth=2,font=dict(color='white'),row=1, col=2)

    max_daily_death_date = global_daily.loc[global_daily['Daily_death'].idxmax(), 'Date']
    max_daily_death = global_daily['Daily_death'].max()
    max_daily_case_value = global_daily['Daily_cases'].iloc[global_daily['Daily_death'].idxmax()]
    fig.add_annotation(x=max_daily_death_date,y=max_daily_death,text=f"Peak Daily Deaths<br>Date: {max_daily_death_date.strftime('%Y-%m-%d')}<br>Confirmed: {max_daily_death:,.0f}<br>Deaths: {max_daily_case_value:,.0f}",
    showarrow=False,arrowhead=1,arrowsize=1,arrowwidth=3,arrowcolor="white",ax=20,ay=-30,bgcolor="#d62728",bordercolor="white",borderwidth=2,font=dict(color='white'),row=2, col=2)
    
    return fig

def create_country_analysis_plots(total_per_country_wide, n):
    """Create country analysis bar plots"""
    top_n = total_per_country_wide.head(n).copy()
    top_active = total_per_country_wide.sort_values(by='active', ascending=False).head(n)
    
    # Create subplots
    fig = ps(rows=2, cols=2, 
             subplot_titles=[f'💀 Top {n} Countries by Deaths', f'🟠 Top {n} Countries by Active Cases',
                             f'📈 Death Rate Analysis<br><sub>Top {n} Countries by Active Cases</sub>',f'🔄 Recovery Rate Analysis<br><sub>Top {n} Countries by Active Cases</sub>'],
             specs=[[{"secondary_y": False}, {"secondary_y": False}],[{"secondary_y": False}, {"secondary_y": False}]],horizontal_spacing=0.1, vertical_spacing=0.15)
    
    # Plot 1: Deaths
    fig.add_trace(go.Bar(x=top_n['Country/Province'],y=top_n['death'],orientation='v',
        marker=dict(color='#d62728', line=dict(color='white', width=2),opacity=0.8, pattern_shape="/", pattern_size=4),
        text=[f"{val:,.0f}" for val in top_n['death']], textposition='outside',textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{x}</b><br>' + '💀 Deaths: %{customdata[0]:,}<br>' + '📊 Total Cases: %{customdata[1]:,}<br>' + '💚 Recovered: %{customdata[2]:,}<br>' + 
        '� Active: %{customdata[3]:,}<br>' + '�📈 Death Rate: %{customdata[4]:.2f}%<br>' + '🔄 Recovery Rate: %{customdata[5]:.2f}%<br><extra></extra>',
        customdata=np.column_stack((top_n['death'], top_n['confirmed'], top_n['recovered'], top_n['active'], top_n['death_rate'],top_n['recovery_rate'])),name='Deaths'),row=1, col=1)
    # Plot 2: Active cases
    fig.add_trace(go.Bar(x=top_active['Country/Province'],y=top_active['active'],orientation='v',
        marker=dict(color='#ff7f0e', line=dict(color='white', width=2),opacity=0.8, pattern_shape="\\", pattern_size=4),text=[f"{val:,.0f}" for val in top_active['active']], textposition='outside',textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{x}</b><br>' + '🟠 Active Cases: %{customdata[0]:,}<br>' + '📈 Death Rate: %{customdata[1]:.2f}%<br>' +
        '🔄 Recovery Rate: %{customdata[2]:.2f}%<br>' + '📊 Avg Daily Cases: %{customdata[3]:.0f}<br>' + '💀 Avg Daily Deaths: %{customdata[4]:.0f}<br><extra></extra>',
        customdata=np.column_stack((top_active['active'],top_active['death_rate'],top_active['recovery_rate'], top_active['average_daily_cases'], top_active['average_daily_deaths'])),name='Active Cases'), row=1, col=2)
    # Plot 3: Death rate
    fig.add_trace(go.Bar(x=top_active['Country/Province'],y=top_active['death_rate'],orientation='v',
        marker=dict(color='#d62728', line=dict(color='white', width=2),opacity=0.8, pattern_shape=".",
                    pattern_size=8),text=[f"{val:.1f}%" for val in top_active['death_rate']], textposition='outside',textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{x}</b><br>' + '📈 Death Rate: %{customdata[1]:.2f}%<br>' + '🟠 Active Cases: %{customdata[0]:,}<br>' +
        '🔄 Recovery Rate: %{customdata[2]:.2f}%<br>' + '📊 Avg Daily Cases: %{customdata[3]:.0f}<br>' + '💀 Avg Daily Deaths: %{customdata[4]:.0f}<br><extra></extra>',
        customdata=np.column_stack((top_active['active'],top_active['death_rate'],top_active['recovery_rate'], top_active['average_daily_cases'], top_active['average_daily_deaths'])), name='Death Rate'), row=2, col=1)
    # Plot 4: Recovery rate
    fig.add_trace(go.Bar(x=top_active['Country/Province'],y=top_active['recovery_rate'],orientation='v',
        marker=dict(color='#2ca02c', line=dict(color='white', width=2),opacity=0.8, pattern_shape="+", pattern_size=8),
        text=[f"{val:.1f}%" for val in top_active['recovery_rate']], textposition='outside',textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{x}</b><br>' + '🔄 Recovery Rate: %{customdata[2]:.2f}%<br>' + '🟠 Active Cases: %{customdata[0]:,}<br>' +
        '📈 Death Rate: %{customdata[1]:.2f}%<br>' + '📊 Avg Daily Cases: %{customdata[3]:.0f}<br>' + '💀 Avg Daily Deaths: %{customdata[4]:.0f}<br><extra></extra>',
        customdata=np.column_stack((top_active['active'],top_active['death_rate'],top_active['recovery_rate'], top_active['average_daily_cases'], top_active['average_daily_deaths'])),name='Recovery Rate'), row=2, col=2)
    
    # Enhanced layout
    fig.update_layout(height=800, width=1200, showlegend=False,plot_bgcolor='rgb(30, 30, 30)', paper_bgcolor='rgb(30, 30, 30)',
                      font=dict(color='white'),title={'text': f'📊 COVID-19 Country Analysis Dashboard', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 24, 'color': 'white'}})
    fig.update_xaxes(showgrid=False, tickfont=dict(size=11, color='white'), tickangle=45)
    fig.update_yaxes(showgrid=True, gridcolor='rgb(100, 100, 100)', gridwidth=0.5, griddash='dash', tickfont=dict(size=13, color='white'), tickformat=',.0f')
    
    return fig

def create_map_visualization(datalong, selected_metric, enable_animation=False, show_daily_cases=False):
    """Create interactive map visualization with optional animation"""
    # Prepare data for map
    map_data = datalong.copy()
    if show_daily_cases:
        # Use daily cases data
        daily_data = map_data[map_data['Status'] == selected_metric].copy()
        pivot_data = daily_data.pivot_table(index=['Country/Province', 'Date', 'Lat', 'Long'], columns=[], values='Daily_cases', fill_value=0, aggfunc='sum').reset_index()
        pivot_data[selected_metric] = pivot_data['Daily_cases']
        
        # Add other metrics as zero for consistency in hover text
        for col in ['confirmed', 'death', 'recovered', 'active']:
            if col != selected_metric:
                pivot_data[col] = 0
    else:
        pivot_data = map_data.pivot_table(
            index=['Country/Province', 'Date', 'Lat', 'Long'], 
            columns='Status', values='Cases', fill_value=0).reset_index()
        
        required_cols = ['confirmed', 'death', 'recovered', 'active']
        for col in required_cols:
            if col not in pivot_data.columns:
                if col == 'active':
                    pivot_data[col] = (pivot_data.get('confirmed', 0) - pivot_data.get('death', 0) - pivot_data.get('recovered', 0)).clip(lower=0)
                else:
                    pivot_data[col] = 0
    
    # Remove rows with no location data
    pivot_data = pivot_data.dropna(subset=['Lat', 'Long'])
    # Color mapping
    color_map = {'confirmed': '#1f77b4','death': '#d62728', 'recovered': '#2ca02c','active': '#ff7f0e'}
    if enable_animation:
        # Get monthly data for animation
        pivot_data['YearMonth'] = pivot_data['Date'].dt.to_period('M')
        monthly_data = pivot_data.groupby(['Country/Province', 'YearMonth', 'Lat', 'Long']).agg({'confirmed': 'max','death': 'max', 'recovered': 'max','active': 'max'}).reset_index()
        
        monthly_data['Date'] = monthly_data['YearMonth'].dt.to_timestamp()
        frames = []
        dates = sorted(monthly_data['Date'].unique())
        
        for date in dates:
            frame_data = monthly_data[monthly_data['Date'] == date]
            if frame_data.empty or frame_data[selected_metric].sum() == 0:
                continue
                
            max_val = frame_data[selected_metric].max()
            min_val = frame_data[selected_metric][frame_data[selected_metric] > 0].min()
            
            if pd.isna(max_val) or pd.isna(min_val) or max_val <= 0:
                normalized_size = np.full(len(frame_data), 10)
            elif max_val > min_val:
                normalized_size = ((np.log10(frame_data[selected_metric] + 1) - np.log10(min_val + 1)) / 
                                  (np.log10(max_val + 1) - np.log10(min_val + 1)) * 25 + 5)
                normalized_size = np.where(np.isnan(normalized_size), 10, normalized_size)
                normalized_size = np.where(normalized_size < 5, 5, normalized_size)
            else:
                normalized_size = np.full(len(frame_data), 10)
            
            hover_text = [f"<b>{country}</b><br>" + f"Date: {date.strftime('%B %Y')}<br>" +f"Confirmed: {confirmed:,}<br>" + f"Deaths: {deaths:,}<br>" + f"Recovered: {recovered:,}<br>" + f"Active: {active:,}<br>"
                         for country, confirmed, deaths, recovered, active in 
                         zip(frame_data['Country/Province'], frame_data['confirmed'],frame_data['death'], frame_data['recovered'], frame_data['active'])]
            
            frame_traces = [go.Scattergeo(lon=frame_data['Long'], lat=frame_data['Lat'],text=hover_text, mode='markers',
                                          marker=dict(size=normalized_size,color=color_map[selected_metric],opacity=0.6,sizemode='diameter',line=dict(width=1, color='white')),
                                          name=selected_metric.title(),hovertemplate='%{text}<extra></extra>')]
            
            frames.append(go.Frame(data=frame_traces,name=date.strftime('%Y-%m-%d'),layout=go.Layout(title_text=f"COVID-19 Global Status - {date.strftime('%B %Y')}")))
        
        # Create figure with animation
        fig = go.Figure(data=frames[0].data if frames else [], frames=frames)
        title_prefix = "Daily" if show_daily_cases else "Cumulative"
        fig.update_layout(
            title={'text': f'Interactive COVID-19 World Map - {title_prefix} {selected_metric.title()} Cases',
                   'x': 0.5, 'font': {'size': 24, 'color': 'white'}},
            geo=dict(projection_type='natural earth',showland=True, landcolor='rgb(50, 50, 50)',coastlinecolor='rgb(100, 100, 100)',showocean=True, 
                     oceancolor='rgb(20, 20, 20)',showlakes=True, lakecolor='rgb(20, 20, 20)',showcountries=True, countrycolor='rgb(100, 100, 100)',bgcolor='rgb(30, 30, 30)'),
            paper_bgcolor='rgb(30, 30, 30)',plot_bgcolor='rgb(30, 30, 30)',font=dict(color='white'),
            legend=dict(bgcolor='rgba(30, 30, 30, 0.8)',bordercolor='rgb(100, 100, 100)',borderwidth=1,font=dict(color='white'),x=0.02, y=0.98),
            width=1200, height=700,
            updatemenus=[{'type': 'buttons','showactive': False,'y': 0.2,'x': 0.05,'xanchor': 'left','yanchor': 'bottom','pad': {'t': 10, 'r': 10},
                'buttons': [{'label': 'Play','method': 'animate','args': [None, {'frame': {'duration': 500, 'redraw': True},'fromcurrent': True,'transition': {'duration': 200}}]},
                    {'label': 'Pause','method': 'animate','args': [[None], {'frame': {'duration': 0, 'redraw': False},'mode': 'immediate','transition': {'duration': 0}}]}]}],
            sliders=[{'active': 0,'yanchor': 'bottom','xanchor': 'center','currentvalue': {'font': {'size': 16, 'color': 'white'},'prefix': 'Date: ','visible': True,'xanchor': 'center'},
                'pad': {'b': 10, 't': 10},'len': 0.9,'x': 0.5,'y': 0.05,'steps': [{'args': [[frame.name], {'frame': {'duration': 300, 'redraw': True},'mode': 'immediate','transition': {'duration': 200}}],
                    'label': pd.to_datetime(frame.name).strftime('%b %Y'),'method': 'animate'} for frame in frames]}])
        
    else:
        latest_data = pivot_data.groupby('Country/Province').last().reset_index()
        latest_data = latest_data[latest_data[selected_metric] > 0]
        
        if latest_data.empty:
            st.warning(f"No data available for {selected_metric}")
            return None  
        max_cases = latest_data[selected_metric].max()
        min_cases = latest_data[selected_metric][latest_data[selected_metric] > 0].min()
        
        if pd.isna(max_cases) or pd.isna(min_cases) or max_cases <= 0:
            normalized_size = np.full(len(latest_data), 10)
        elif max_cases > min_cases:
            normalized_size = ((np.log10(latest_data[selected_metric] + 1) - np.log10(min_cases + 1)) / 
                              (np.log10(max_cases + 1) - np.log10(min_cases + 1)) * 25 + 5)
            normalized_size = np.where(np.isnan(normalized_size), 10, normalized_size)
            normalized_size = np.where(normalized_size < 5, 5, normalized_size)
        else:
            normalized_size = np.full(len(latest_data), 10)
        
        hover_text = [f"<b>{country}</b><br>" + f"Confirmed: {confirmed:,}<br>" + f"Deaths: {deaths:,}<br>" + f"Recovered: {recovered:,}<br>" +f"Active: {active:,}<br>"
                     for country, confirmed, deaths, recovered, active in 
                     zip(latest_data['Country/Province'], latest_data['confirmed'],latest_data['death'], latest_data['recovered'], latest_data['active'])]
        
        # Create static map
        fig = go.Figure()
        title_prefix = "Daily" if show_daily_cases else "Cumulative"
        fig.add_trace(go.Scattergeo(lon=latest_data['Long'], lat=latest_data['Lat'],text=hover_text, mode='markers',
            marker=dict(size=normalized_size,color=color_map[selected_metric],opacity=0.6,sizemode='diameter',
                        line=dict(width=1, color='white')),name=selected_metric.title(),hovertemplate='%{text}<extra></extra>'))
        
        fig.update_layout(
            title={'text': f'COVID-19 World Map - {title_prefix} {selected_metric.title()} Cases','x': 0.5, 'font': {'size': 24, 'color': 'white'}},
            geo=dict(projection_type='natural earth',showland=True, landcolor='rgb(50, 50, 50)',coastlinecolor='rgb(100, 100, 100)',showocean=True, oceancolor='rgb(20, 20, 20)',showlakes=True, 
                     lakecolor='rgb(20, 20, 20)',showcountries=True, countrycolor='rgb(100, 100, 100)',bgcolor='rgb(30, 30, 30)'),paper_bgcolor='rgb(30, 30, 30)',plot_bgcolor='rgb(30, 30, 30)',font=dict(color='white'),width=1200, height=700)
    
    return fig

# Main app
def main():
    st.title("🦠 COVID-19 Data Visualization Dashboard")
    st.markdown("---")
    # Load data
    with st.spinner('Loading data...'):
        result = load_data()
    if result is None:
        st.error("Failed to load data. The load_data() function returned None.")
        st.error("Please check if the data files exist in the ../Data/ directory or run the data_cleaning.ipynb notebook first.")
        return
        
    datalong, total_per_country_wide = result
    if datalong is None or total_per_country_wide is None:
        st.error("Failed to load data. Please check if the data files exist in the ../Data/ directory.")
        return
    
    st.sidebar.header("🎛️ Dashboard Controls")
    countries = get_country_list(datalong)
    selected_country = st.sidebar.selectbox("🌍 Select Country/Region:",options=countries,index=0,  help="Type to search for a specific country")
    # Top N selection for country analysis
    n_countries = st.sidebar.slider("📊 Number of countries to show:",min_value=3,max_value=25,value=5,help="Select how many top countries to display in the analysis")
    # Map metric selection
    map_metric = st.sidebar.selectbox("🗺️ Map Visualization Metric:",options=['confirmed', 'death', 'recovered', 'active'],index=3,help="Select which metric to display on the world map")
    # Animation toggle
    enable_animation = st.sidebar.checkbox("🎬 Enable Map Animation",value=True,help="Enable time-based animation on the map (may take longer to load)")
    # Daily cases toggle
    show_daily_cases = st.sidebar.checkbox("📊 Daily Cases",value=True,help="Toggle between cumulative and daily cases on the map")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎨 Color Legend")
    st.sidebar.markdown("🔵 **Blue**: Confirmed Cases")
    st.sidebar.markdown("🔴 **Red**: Deaths") 
    st.sidebar.markdown("🟢 **Green**: Recovered* Cases")
    st.sidebar.markdown("🟠 **Orange**: Active Cases")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("💡 **Tip**: Run `data_cleaning.ipynb` to generate processed data for faster app startup!")
    
    # Calculate metrics
    metrics = calculate_metrics(datalong, total_per_country_wide, selected_country)
    if metrics is None:
        st.error(f"No data found for {selected_country}")
        return
    
    # Display metrics
    st.header(f"📈 COVID-19 Metrics: {selected_country}")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Total Confirmed</div>
            <div class="metric-value" style="color: #1f77b4;">{metrics['confirmed_total']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Total Deaths</div>
            <div class="metric-value" style="color: #d62728;">{metrics['deaths_total']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Total Recovered*</div>
            <div class="metric-value" style="color: #2ca02c;">{metrics['recovered_total']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Total Active</div>
            <div class="metric-value" style="color: #ff7f0e;">{metrics['active_total']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Death Rate</div>
            <div class="metric-value" style="color: #d62728;">{metrics['death_rate']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent data metrics
    st.subheader("📅 Last 7 Days")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("New Confirmed", f"{metrics['recent_confirmed']:,}")
    with col2:
        st.metric("New Deaths", f"{metrics['recent_deaths']:,}")
    with col3:
        st.metric("New Recovered", f"{metrics['recent_recovered']:,}")
    with col4:
        st.metric("New Active", f"{metrics['recent_active']:,}")
    with col5:
        st.metric("Global Rank", f"#{metrics['rank']}")
    
    st.markdown("---")
    # Time series plots
    st.header("📊 Time Series Analysis")
    with st.spinner('Creating comprehensive time series plots...'):
        time_series_fig = create_time_series_plots(datalong, selected_country, total_per_country_wide)
        st.plotly_chart(time_series_fig, use_container_width=True)
    
    st.markdown("---")
    # Country analysis plots
    st.header("🌍 Country Analysis")
    with st.spinner('Creating country analysis plots...'):
        country_fig = create_country_analysis_plots(total_per_country_wide, n_countries)
        st.plotly_chart(country_fig, use_container_width=True)
    
    # Display ranking table
    st.subheader("Country Rankings")
    top_countries = total_per_country_wide.head(n_countries)[['Country/Province', 'confirmed', 'death', 'recovered', 'active', 'death_rate', 'recovery_rate', 'Rank']]
    top_countries.columns = ['Country/Province', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Death Rate (%)', 'Recovery Rate (%)', 'Rank']
    st.dataframe(top_countries, use_container_width=True)
    
    st.markdown("---")
    # Map visualization
    st.header("🗺️ Interactive World Map")
    with st.spinner('Creating map visualization...'):
        map_fig = create_map_visualization(datalong, map_metric, enable_animation, show_daily_cases)
        if map_fig:
            st.plotly_chart(map_fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("### 📝 Data Sources")
    st.markdown("Data sourced from Johns Hopkins University COVID-19 dataset via Kaggle")
    st.markdown("#### ***Recovery data for some countries (e.g., US/Canada) may be underestimated due to limited or inconsistent reporting from original sources**")
    #Gituib
    st.markdown("---")
    st.markdown("📁 The full open-source project is hosted on my personal [GitHub](https://github.com/Papagiannopoulos)")

if __name__ == "__main__":
    main()