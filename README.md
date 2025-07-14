# 🦠 **COVID-19 Globe Dashboard** 

🎯 ***Aim***: Visualize and analyze COVID-19 data (confirmed/recovered/active cases & deaths) worldwide through interactive web application built with **Streamlit**. 

## Project Summary 
This dashboard provides (real-time) insights into the pandemic's progression overtime from Jan/20 till Mar/21.

## Project Structure

### Table of Contents
1. 🔍 **[ Dataset](#-dataset)** - Data source
2. 🧹 **[ Data Processing or Methods](#-data-processing)** - Cleaning and feature engineering
3. 📊 **[ EDA](#-eda)** - Key insights and patterns
4a. ▶️ **[ Insights/Results 1](#-insights/Results-1)** - text
4b. ▶️ **[ Insights/Results 2](#-insights/Results-2)** - text
4c. ▶️ **[ Insights/Results 3](#-insights/Results-3)** - text
5. **[ Discussion](#Discussion)** - Next steps and improvements
6. **[ Findngs Implications/Conclusions](#-findngs-Implications/Conclusions)** - Next steps and improvements
7. **[ Future Enhancements](#-future-enhancements)** - Next steps and improvements
8. 🔁 **[ Reproducibility](#-reproducibility)** - Reproducibility steps

## 🔍 Dataset 
I used the Kaggle API (Program > kaggle_data_download.py) to automatically load the dataset.  
Navigate to the **[ Reproducibility](#-reproducibility)** for more details regarding Kaggle API.  
More info about data is available at the following link.  
[📥 Download Source](https://www.kaggle.com/datasets/sudalairajkumar/novel-corona-virus-2019-dataset)

## 🧹Data Processing
 
 
## 📊 EDA


## ▶️ Insights/Results 1


## ▶️ Insights/Results 2


## ▶️ Insights/Results 3


## ▶️ Discussion


## ▶️ Findngs Implications/Conclusions


## ▶️ Future Enhancements


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