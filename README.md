# Mental Health EDA: Global Trends Analysis (1990-2019)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mental-health-eda-webpage.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive exploratory data analysis (EDA) platform that visualizes three decades of global mental health trends, uncovering patterns in prevalence and disease burden across countries and continents.

## 🌍 Overview

Mental health is one of the most critical challenges of our time, yet understanding its global patterns requires diving deep into complex datasets. This project analyzes mental health disorder data from 1990 to 2019, providing interactive visualizations to explore how the landscape has evolved worldwide.

**Live Dashboard:** [mental-health-eda-webpage.streamlit.app](https://mental-health-eda-webpage.streamlit.app/)

## ✨ Key Features

### 📊 Interactive Visualizations
- **Choropleth Maps**: Year-by-year geographical visualization of mental health disorder prevalence
- **Temporal Analysis**: Track evolution of disorders across three decades
- **Comparative Analytics**: Global vs. regional statistics with continent and country-level breakdowns
- **Disease Burden Metrics**: Analysis of DALYs (Disability-Adjusted Life Years) for nuanced quality of life impact assessment

### 🎯 Analysis Capabilities
- Identification of trends in developing vs. developed nations
- Detection of statistical outliers and regional anomalies
- Multi-dimensional filtering and real-time data exploration
- Comprehensive demographic comparisons

## 🛠️ Tech Stack

- **Analysis & Processing**: Python, Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Web Application**: Streamlit
- **Data Source**: 
  - Global Burden of Disease (GBD) study by Institute for Health Metrics and Evaluation (IHME), University of Washington
  - World Health Organization (WHO)

## 🚀 Getting Started

### Prerequisites
```bash
python >= 3.8
```

### Installation

1. Clone the repository
```bash
git clone https://github.com/LunaPerezT/Mental-Health-EDA-Webpage.git
cd Mental-Health-EDA-Webpage
```

2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

### Running Locally

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📁 Project Structure

```
Mental-Health-EDA-Webpage/
├── .streamlit/           # Streamlit configuration files
├── Graphs/               # Generated graphs and visualizations
├── data/                 # Dataset files
├── docs/                 # Documentation files
├── img/                  # Images and assets
├── notebooks/            # Jupyter notebooks for analysis
├── static/               # Static files for web application
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 📈 Key Insights

- **Geospatial Patterns**: Visualization of how mental health disorder prevalence varies geographically and temporally
- **Burden Analysis**: Beyond prevalence, understanding the actual impact on quality of life through DALY metrics
- **Regional Disparities**: Identification of significant differences between developing and developed nations
- **Trend Detection**: Multi-decade patterns revealing shifts in global mental health landscape

## 🔍 Data Sources

This project utilizes authoritative global health data from:
- **IHME Global Burden of Disease (GBD)**: Comprehensive health metrics from the University of Washington
- **World Health Organization (WHO)**: International health statistics and standardized definitions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Luna Perez**
- GitHub: [@LunaPerezT](https://github.com/LunaPerezT)
- Project Link: [https://github.com/LunaPerezT/Mental-Health-EDA-Webpage](https://github.com/LunaPerezT/Mental-Health-EDA-Webpage)

## 🙏 Acknowledgments

- Institute for Health Metrics and Evaluation (IHME) for the GBD dataset
- World Health Organization (WHO) for supplementary data
- The open-source community for the amazing tools that made this project possible

## 📧 Contact

For questions or feedback, please open an issue in the GitHub repository.

---

**Note**: This project is for educational and research purposes. Mental health data should be interpreted by qualified professionals for clinical or policy decisions.
