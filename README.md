# 🦠 **COVID-19 Globe Dashboard** 
#### 🎯 ***Aim***: Visualize and analyze daily level COVID-19 cases worldwide through interactive web application built with **Streamlit**. <img src="https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png" width="60"/>  

## Project Summary 
This dashboard provides (real-time) insights into the pandemic's progression overtime from Jan/20 till May/21.
- **Comprehensive Coverage**: Data from 180+ countries and regions
- **Comparative Analysis**: Side-by-side country comparisons
- **Historical Trends**: Track pandemic evolution over time

## Project Structure

### Table of Contents
1. 🔍 **[ Dataset](#-dataset)** - Data source  
2. 🧹 **[ Data Processing or Methods](#data-processing)** - Cleaning and feature engineering  
3. 📊 **[ EDA](#-eda)** - Key insights and patterns  
4. <img src="https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png" width="60"/> **[ Streamlit app](#-streamlit-app)** - Dashboard features  
5. ▶️ **[ Insights/Results 1](#%EF%B8%8F-insightsresults-1)** - text  
6. 🔍**[ Findngs Implications/Conclusions](#-findngs-Implications/Conclusions)** - Next steps and improvements  
7. 🚀 **[ Future Enhancements](#-future-enhancements)** - Next steps and improvements  
8. 🔁 **[ Reproducibility](#-reproducibility)** - Reproducibility steps  

## 🔍 Dataset 
I used the Kaggle API (Program > kaggle_data_download.py) to automatically load the dataset.  
Navigate to the **[ Reproducibility](#-reproducibility)** for more details regarding Kaggle API.  
More info about data is available at the following link.  
	[📥 Download Source](https://www.kaggle.com/datasets/sudalairajkumar/novel-corona-virus-2019-dataset)

## 🧹Data Processing
1) **Data Cleaning**: geographical data aggegations, removed duplicates (<1%), missing values handling  
2) **Feature Engineering**: Created active cases, daily/comulative cases, date components, metrics (death/recovery rate, overall, date dependent) for visualisations  

## <img src="https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png" width="40"/> Streamlit app
#### 🛠️ Technology Stack
- **Visualizations**: Plotly (interactive charts and maps)
- **Data Processing**: Pandas, NumPy

#### 📈 Key Metrics Tracked
- Total Confirmed/Recovered*/Active Cases
- Total Deaths
- Death/Recovery Rate
- Recent 7-day trends
- Global Rankings

#### 📊 Interactive Visualizations
- **Real-time Metrics**: Key performance indicators with recent trends
- **Time Series Analysis**: Dynamic 2x2 subplot layout showing cumulative and daily cases/deaths
- **Country Analysis**: Comparative bar charts for top countries by various metrics
- **Interactive World Map**: Animated geographical visualization with time-based controls

## 📊 Dashboard Sections

#### 1. Metrics Overview
- Key statistics displayed in visually appealing metric cards
- Recent 7-day trends with change indicators
- Global ranking information

#### 2. Time Series Analysis
- 2x2 subplot layout matching notebook visualizations
- Cumulative and daily cases/deaths
- Peak annotations with detailed information
- Interactive zoom and pan capabilities

#### 3. Country Analysis
- Top N countries by deaths and active cases
- Death rate and recovery rate analysis
- Detailed hover information and rankings table

#### 4. Interactive World Map
- Geographic visualization with circle markers
- Size proportional to case numbers
- Time-based animation controls
- Toggle between cumulative and daily cases
- Color-coded by metric type

## 📊 EDA


## ▶️ Insights/Results 1


## 🔍 Findngs Implications/Conclusions
### Data-Driven Decision Making
- **Public Health Officials**: Monitor pandemic progression and identify hotspots
- **Researchers**: Analyze trends and patterns across different regions
- **Policymakers**: Make informed decisions based on current data
- **General Public**: Stay informed about COVID-19 status in their region

## 🚀 Future Enhancements

#### 📈 Enhanced Analytics
- Predictive modeling and forecasting
- Machine learning integration for trend analysis
- Advanced statistical indicators and correlations

#### 🌍 Extended Data Sources
- Integration with multiple health organizations
- Real-time API connections
- Additional population characteristics (e.g. popoulation sizes)

## 🔁 Reproducibility
#### 1. Clone repo and cd
git clone https://github.com/Papagiannopoulos/who-covid19-globe-dashboard.git   
cd 'ecommerce-business-analytics'

#### 2. Create a fresh virtual [env](https://github.com/astral-sh/uv)
uv venv  
**Note**: If uv is not already installed, run the following command in PowerShell.  
- On macOS and Linux:  
curl -LsSf https://astral.sh/uv/install.sh | sh  
- On Windows:  
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

#### 3. Sync environment
uv sync  
**Note**: At this step, Microsoft Visual C++ is required. If sync crashes follow the provided steps.

#### 4. Kaggle's API  
1) Create a [Kaggle account](https://www.kaggle.com)  
2) Go to Account Settings and click "Create New API Token" to download the kaggle.json file  
3) Navigate to C:\Users\<your_user_name> on your computer  
4) Create a new folder named .kaggle  
5) Move the downloaded kaggle.json file into the .kaggle folder

#### 5. You are ready!!!