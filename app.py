# STREAMLIT MENTAL HEALTH EDA WEB PAGE
# Luna Pérez Troncoso

#-----------LIBRARIES LOADING-------------

import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as snss
import folium
import matplotlib as mpl
import streamlit_folium as sf
import branca.colormap as cm
from folium import plugins
import plotly
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go

#-----------DATAFRAME LOADING-------------

df_mh=pd.read_csv("./data/1-mental-illnesses-prevalence.csv")
df_mh_burden=pd.read_csv("./data/2-burden-disease-from-each-mental-illness.csv")
df_mh.columns=["country","code","year","Schizophrenia","Depressive","Anxiety","Bipolar","Eating"]
df_mh_income = df_mh[(df_mh.country =='Low-income countries')|(df_mh.country =='Lower-middle-income countries')|(df_mh.country =='Upper-middle-income countries')|(df_mh.country =='High-income countries')]
df_mh.drop(df_mh.index[(df_mh.country =='Low-income countries')|(df_mh.country =='Lower-middle-income countries')|(df_mh.country =='Upper-middle-income countries')|(df_mh.country =='High-income countries')],axis=0,inplace=True)
df_mh_world=df_mh[df_mh.country =="World"]
df_mh.drop(df_mh.index[(df_mh.country =='European Union (27)')|(df_mh.country =="World")],inplace=True)
df_mh_cont = df_mh[(df_mh.country =="Asia (IHME GBD)")|(df_mh.country =='Europe (IHME GBD)')|(df_mh.country =='America (IHME GBD)')|(df_mh.country =='Africa (IHME GBD)')]
df_mh.drop(df_mh.index[(df_mh.country =="Asia (IHME GBD)")|(df_mh.country =='Europe (IHME GBD)')|(df_mh.country =='America (IHME GBD)')|(df_mh.country =='Africa (IHME GBD)')],axis=0,inplace=True)
df_mh_burden.columns=["country","code","year","Schizophrenia","Depressive","Anxiety","Bipolar","Eating"]
df_mh_burden_income = df_mh_burden[(df_mh_burden.country == 'Low income (WB)')|(df_mh_burden.country == "Lower middle income (WB)")|(df_mh_burden.country =='Middle income (WB)')|(df_mh_burden.country =='High income (WB)')]
df_mh_burden.drop(df_mh_burden.index[(df_mh_burden.country == 'Low income (WB)')|(df_mh_burden.country == "Lower middle income (WB)")|(df_mh_burden.country =='Middle income (WB)')|(df_mh_burden.country =='High income (WB)')],axis=0,inplace=True)
df_mh_burden_WB = df_mh_burden[(df_mh_burden.country =='Europe & Central Asia (WB)')|(df_mh_burden.country =='East Asia & Pacific (WB)')|(df_mh_burden.country =='Latin America & Caribbean (WB)')|(df_mh .country =='Middle East & North Africa (WB)')|(df_mh_burden.country =='North America (WB)')|(df_mh_burden.country =='South Asia (WB)')|(df_mh_burden.country =='Sub-Saharan Africa (WB)')]
df_mh_burden.drop(df_mh_burden.index[(df_mh_burden.country =='Europe & Central Asia (WB)')|(df_mh_burden.country =='East Asia & Pacific (WB)')|(df_mh_burden.country =='Latin America & Caribbean (WB)')|(df_mh_burden.country =='Middle East & North Africa (WB)')|(df_mh_burden.country =='North America (WB)')|(df_mh_burden.country =='South Asia (WB)')|(df_mh_burden.country =='Sub-Saharan Africa (WB)')],axis=0,inplace=True)
df_mh_burden_world=df_mh_burden[df_mh_burden.country =="World"]
df_mh_burden_cont = df_mh_burden[(df_mh_burden.country =='Western Pacific Region (WHO)')|(df_mh_burden.country =='Eastern Mediterranean Region (WHO)')|(df_mh_burden.country =='European Region (WHO)')|(df_mh_burden.country =='Region of the Americas (WHO)')|(df_mh_burden.country =='African Region (WHO)')|(df_mh_burden.country =='South-East Asia Region (WHO)')]
df_mh_burden.drop(df_mh_burden.index[(df_mh_burden.country =='Western Pacific Region (WHO)')|(df_mh_burden.country =='Eastern Mediterranean Region (WHO)')|(df_mh_burden.country =='European Region (WHO)')|(df_mh_burden.country =='Region of the Americas (WHO)')|(df_mh_burden.country =='African Region (WHO)')|(df_mh_burden.country =='South-East Asia Region (WHO)')|(df_mh_burden.country =="World")|(df_mh_burden.country =='OECD Countries')],axis=0,inplace=True)

df_mh_c_y=df_mh.iloc[:,[0,2,3,4,5,6,7]].groupby(["country","year"]).mean()
df_mh_burden_c_y=df_mh_burden.iloc[:,[0,2,3,4,5,6,7]].groupby(["country","year"]).mean()


df_mh_mean=df_mh.iloc[:,1:].groupby("code").mean().iloc[:,1:]
df_mh_mean.reset_index(inplace=True,names=["code"])
df_mh_burden_mean=df_mh_burden.iloc[:,1:].groupby("code").mean().iloc[:,1:]
df_mh_burden_mean.reset_index(inplace=True,names=["code"])
df_mh_world_mean=df_mh_world.drop(["year","code"],axis=1).set_index("country").groupby("country").mean()
df_mh_mean2=df_mh.drop(["year","code"],axis=1).set_index("country").groupby("country").mean()
df_mh_burden_world_mean=df_mh_burden_world.drop(["year","code"],axis=1).set_index("country").groupby("country").mean()
df_mh_burden_mean2=df_mh_burden.drop(["year","code"],axis=1).set_index("country").groupby("country").mean()
fname = './data/countries.geo.json'
geo = gpd.read_file(fname)

# ---------- PLOT COLORS SETTING ----------

colors = ['#636EFA', '#FFA15A', '#00CC96', '#EF553B', '#AB63FA']

# ---------- PAGE CONFIGURATION ----------

st.set_page_config(page_title="Global Mental Health EDA",page_icon="🧠",layout="wide",initial_sidebar_state="expanded")

# ---------- CUSTOM CSS STYLING ----------

st.markdown(
"""
    <style>
    /* SELECTBOX */
    .stSelectbox label {
        font-size: 16px !important;
        font-weight: 500 !important;
        color: #391550 !important;
    }
    /* INFO BOX COLOR CUSTOMIZATION */
    div[data-testid="stInfo"] {
        border-left: 6px solid #0747d4 !important;
        background-color: #F0F2F6 !important;
        color: #391550 !important;
    }
    /* Custom highlight box (placeholder) */
    .highlight-box {
        border-left: 6px solid #F0F2F6;
        background-color: #FFFFFF;
        color: black;
        padding: 12px 16px;
        border-radius: 20px;
        margin: 5px 5px;
        font-size: 15px;;
    }

    /* Author analysis box */
    .author-box {
        border-left: 6px solid #0747d4;
        background-color: #F0F2F6;
        color: black;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 14.5px;
    }

    /* Custom danger box */
    .danger-box {
        border-left: 6px solid #FFA500;
        background-color: #F9D590;
        color: black;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 5px 5px;
        font-size: 15px;;
    }

    </style>
    """
  ,unsafe_allow_html=True)

# ---------- APP HEADER ----------

st.markdown('''<h1 style="text-align: center;color: black;"> <b>Global Mental Health EDA Dashboard</b></h1>
    <h5 style="text-align: center;color: gray"> Luna Pérez Troncoso </h5>''', unsafe_allow_html=True)
st.markdown("---")

# ---------- SIDEBAR NAVIGATION ----------

st.sidebar.title("Navigation Menu")
page = st.sidebar.radio(
    "Select Section:",
    (
        "🏠 Home",
        "📚 Introduction",
        "🗐 Data Description and Sources",
        "📂 Raw Data",
        "📊 Global Statistics",
        "🌐 Cross-Country Distributions",
        "🔗 Disorders' Metrics Correlations",
        "🌍 Global Choropleth Maps",
        "🕰️ Comparison of Choropleth Maps by Year",
        "✍️ Conclusions",
        "🙋🏻‍♀️ About the Author"
    ),index=0
)

def future():
    ''' Future add:         "📈 General Cross-Country Progressions",
        "🗺️ Geographic Regions Based Analysis",
        "💰 Income-Level Based Analysis",
        "🚩 Outlying Countries Visualization"
    '''

def highlight_box(text: str):
    html = f'''<div class="highlight-box">
    
    🔹 {text}
    
    </div>'''
    st.markdown(html, unsafe_allow_html=True)

def author_box(md_text: str):
    html = f'<div class="author-box">{md_text}</div>'
    st.markdown(html, unsafe_allow_html=True)


# ---------- SECTION TEMPLATE ----------
def section_with_selectbox(title, chart_options):
    st.header(title)

    # selectone (single choice)
    selected = st.selectbox("Select graphical visualization:", chart_options, index=0)

    # Subheader and custom highlight box (uses #391550 accent)
    st.subheader(selected)
    highlight_box(f"Placeholder for **{selected}** graphical visualization.")

    # separator
    st.markdown("<br></br>",unsafe_allow_html=True)

    show_analysis = st.expander("Show Author’s Analysis", expanded=True)
    if show_analysis:
        # You can replace the text below with actual markdown content
        author_md = """
        <h4 style="margin:0 0 8px 0;">✍️ Author’s Analysis</h4>
        <p style="margin:0;">
        This section provides the author's interpretation and concise analytical commentary 
        about the selected graph, highlighting key patterns, anomalies, and suggested next steps.
        </p>
        """
        author_box(author_md)


# ---------- PAGE CONTENTS ----------

if page == "🏠 Home":
    st.header("Welcome to the Global Mental Health EDA")
    st.markdown("""
        This project aims to provide a **comprehensive exploration** of global mental health data, 
        covering major disorders such as anxiety, depression, bipolar, eating disorders, and schizophrenia.

        Through this dashboard, you can:
        - **Explore global statistical summaries**
        - **Visualize distributions** of prevalence and burden across countries
        - **Track long-term trends** across income levels and world regions
        - **Identify outliers and correlations** between disorder metrics
        - **Inspect global and temporal choropleth maps**
        
        ---
        ← *Use the sidebar to select any analytical section and explore the interactive visuals.*
        """)

   
elif page == "📚 Introduction":
    st.header("📚 Introduction")
    st.markdown("""
        **Mental health plays a fundamental role in the overall well-being of individuals and the stability of societies**. It shapes how people think, feel, and act, influencing their relationships, productivity, and quality of life. When mental health deteriorates, **the consequences extend beyond the individual, affecting families, workplaces, and communities at large**.

        Across the globe, mental health disorders represent a **growing public health concern**. Hundreds of millions of people experience conditions such as anxiety and depression every year, and these numbers continue to rise. It is estimated that nearly **one in three women and one in five men will experience a major depressive episode at some point in their lives**. Although disorders like schizophrenia or bipolar disorder are less frequent, their social and economic impact remains profound.

        **This exploratory data analysis (EDA) aims to investigate the global prevalence and burden of mental health disorders** across countries, continents, and socioeconomic regions.By identifying global trends and potential contributing factors, **this analysis seeks to provide a clearer understanding of how mental health challenges manifest across different populations and what patterns may inform prevention and intervention strategies**.
        
        <br/><br/>

        """,unsafe_allow_html=True)


elif page =="🗐 Data Description and Sources":
    st.header("🗐 Data Description and sources")
    st.markdown("""
            The both datasets are composed by **two text columns**. The first one identify the **entities** (country, continent, geographic region or grouped countries data classified by income level), whereas the second one refers to the **year**. Each entity and year has information of the **prevalence and** the **burden** of five relevant **mental health disorders**: **anxiety** disorders, **bipolar** disorders, **depressive** disorders, **eating** disorders and **schizophrenia** spectrum disorders.   
            Both datasets cover the **time period between 1990 and 2019**.
                     
            Both prevalence and burden data are age-standardized and comes from both mixed sex. **Prevalence** data is **expressed in percentages** of prevalence, whereas **burden** is **meassured by Disability-Adjusted Life Years (DALYs) rate per 100.000 Population**. Disability-adjusted life years (DALYs) represent the sum of years lost to premature death and years lived with disability.

        """,unsafe_allow_html=True)
    with st.container(border=True):
        st.image("./img/DALYs.png",use_container_width=True)
    st.markdown("---")
    st.header("Data Sources")
    st.markdown("""
        Global mental health disorders prevalence and burden data were obtained from a [kaggle dataset](https://www.kaggle.com/datasets/imtkaggleteam/mental-health) shared by Mohamadreza Momeni in CSV format.   
                  
        This data was collected throught [OurWorldInData](https://ourworldindata.org/) platform from two main sources: [Global Burden of Disease study by Institute for Health Metrics and Evaluation (GBD-IHME)](https://ghdx.healthdata.org/) from the University of Washington and [World Health Organization (WHO)](https://www.who.int/).

        <br/><br/>
        """,unsafe_allow_html=True)  
    with st.container(border=True):
        col6, col1, col2, col3, col4, col5 = st.columns([1,6,6,6,6,1],gap="large",vertical_alignment="center")
        with col1:
            st.image("./img/kaggle.png",use_container_width=True)
        with col2:
            st.image("./img/OWiD.png",use_container_width=True)
        with col3:
            st.image("./img/IHME-GHDX.png",use_container_width=True)
        with col4:
            st.image("./img/WHO.jpg",use_container_width=True)
    st.markdown("---")
    st.header("Countries Classification Criteria")
    st.markdown("""
        In this section we are going to analyze the **temporal progression and the stacked prevalence and burden data of different geographic regions**, 
        using **line plots and stacked bar plots**. Regional classification was done following two different criteria: **continents in prevalence** data and 
        [**World Health Organization (WHO) geographic regions**](https://ourworldindata.org/grapher/who-regions) or [**The World Bank (WB) geographic regions**](https://datatopics.worldbank.org/world-development-indicators/the-world-by-income-and-region.html) in **burden data**.
        """,unsafe_allow_html=True)  
    show_who = st.expander("Show world regions map according to the World Health Organization", expanded=True)
    with show_who:
        st.image("./img/who-regions.png",use_container_width="auto")
    st.markdown("<br/><br/>", unsafe_allow_html=True)

    st.markdown("""  
        World countries were classified by income level into 4 categories following [The World Bank (WB) criteria](https://datatopics.worldbank.org/world-development-indicators/the-world-by-income-and-region.html). 
        """,unsafe_allow_html=True)   
    show_wb = st.expander("Show world countries classified by level of income according to The World Bank", expanded=True)
    with show_wb:
        st.image("./img/wb-regions.png",use_container_width="auto")
        st.image("./img/leyenda_WB.png",use_container_width="auto")
    st.markdown("<br/><br/>", unsafe_allow_html=True)

elif page == "📂 Raw Data":
    st.markdown(""" 
                <center>

                ### **Mental Heath Disorders Prevalence Raw DataFrame**

                </center>
                """,unsafe_allow_html=True) 
    st.dataframe(pd.read_csv("./data/1-mental-illnesses-prevalence.csv"))
    st.markdown(""" 
                <center>

                ### **Mental Heath Disorders Burden Raw DataFrame**

                </center>
                """,unsafe_allow_html=True) 
    st.dataframe(pd.read_csv("./data/2-burden-disease-from-each-mental-illness.csv"))

elif page == "📊 Global Statistics":
    st.header("📊 Global Statistics")
    with st.container(border=True):
        selected = st.radio("Select graphical visualization:", ["Global Descriptive Statistics of Prevalence by Disorder","Global Descriptive Statistics of Burden by Disorder","Overall Summary Prevalence Statistics","Overall Summary Burden Statistics"], index=0)
    st.subheader(selected)
    if selected == "Global Descriptive Statistics of Prevalence by Disorder":
        st.markdown('''
                    <div class="danger-box">
            
                    ⚠️ It's important to emphasize that **standard deviations do not represent the variation of prevalence and burden through the different countries**. These standard deviations summerize the variation of the global prevalence of these disorders between 1990 and 2019.")
        
                    </div>
                    ''',unsafe_allow_html=True)
        st.markdown("<br/><br/>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center;color: gray'>Age Standarized Global Mental Health Disorders Prevalence (%) Statistics<h5/>",unsafe_allow_html=True)
        st.dataframe(df_mh_world.describe().iloc[1:,1:].T.style.background_gradient(cmap="rocket_r", axis=0),use_container_width=True)
        st.image("./img/bar.png",use_container_width= "always")

    if selected == "Global Descriptive Statistics of Burden by Disorder":
        st.markdown('''
                    <div class="danger-box">
            
                    ⚠️ It's important to emphasize that **standard deviations do not represent the variation of prevalence and burden through the different countries**. These standard deviations summerize the variation of the global prevalence of these disorders between 1990 and 2019.")
        
                    </div>
                    ''',unsafe_allow_html=True)
        st.markdown("<br/><br/>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center;color: gray'>Age Standarized Global Mental Health Disorders DALYs Rate per 100000 Population Statistics<h5/>",unsafe_allow_html=True)
        st.dataframe(df_mh_burden_world.describe().iloc[1:,1:].T.style.background_gradient(cmap="rocket_r", axis=0),use_container_width=True)
        st.image("./img/bar.png",use_column_width= "always")

    if selected == "Overall Summary Prevalence Statistics":
        st.markdown("<br/><br/>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center;color: gray'>Age Standarized Cross-Country Mental Health Disorders Prevalence (%) Statistics<h5/>",unsafe_allow_html=True)
        st.dataframe(df_mh.describe().iloc[1:,1:].T.style.background_gradient(cmap="rocket_r", axis=0),use_container_width=True)
        st.image("./img/bar.png",use_container_width= "always")

    if selected == "Overall Summary Burden Statistics":
        st.markdown("<br/><br/>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center;color: gray'>Age Standarized Cross-Country Mental Health Disorders DALYs Rate per 100000 Population Statistics<h5/>",unsafe_allow_html=True)
        st.dataframe(df_mh_burden.describe().iloc[1:,1:].T.style.background_gradient(cmap="rocket_r", axis=0),use_container_width=True)
        st.image("./img/bar.png",use_container_width= "always")


    # separator
    st.markdown("---")
    show_analysis = st.expander("Show Author’s Analysis", expanded=True)
    with show_analysis:
        st.markdown("""
        <h4 style="margin:0 0 8px 0;">✍️ Author’s Analysis</h4>
                    
        <div class="author-box">  

        <p style="margin:0;">         
        
        When analyzing both prevalence and DALY rates together, a clear pattern emerges:
        - **Depressive and anxiety disorders** are the **most prevalent globally** (≈3.5–3.8%) but show a **moderate individual burden** (≈105–185 DALYs). This suggests that while they are **widespread**, their average **severity per case is lower** compared to less common disorders.
        - In contrast, **schizophrenia and eating disorders** have very **low prevalence** (≈0.16–0.29%) but high DALY rates (≈360–594), indicating a **severe impact on individual** functioning and **quality of life**.
        - **Bipolar disorder** lies in between, with **moderate values in both prevalence (≈0.49%) and burden (≈34 DALYs)**.
        Overall, this combined view highlights two key challenges for global mental health:
        - High-prevalence disorders (depression, anxiety) require broad preventive and treatment coverage due to their societal reach.
        - Low-prevalence but high-burden disorders (schizophrenia, eating disorders) demand specialized and intensive care resources to reduce individual disability.

        </p>
        <div/>                                                     
        """, unsafe_allow_html=True)

        st.markdown("<br/><br/>", unsafe_allow_html=True)

elif page == "🌐 Cross-Country Distributions":
    st.header("🌐 Cross-Country Distributions")
    with st.container(border=True):
        selected = st.radio("Select graphical visualization:", ["Cross-Country Mental Health Disorders Histograms","Cross-Country Mental Health Disorders Distribution Plots","Cross-Country Mental Health Disorders Violin Plots"], index=0)
    st.subheader(selected)
    group_labels = ['Schizophrenia Disorders','Depressive Disorders','Anxiety Disorders', 'Bipolar Disorders','Eating Disorders']
    if selected == "Cross-Country Mental Health Disorders Histograms":
        with st.container(border=True):
            selected_m = st.radio("Select metrics:", ["Prevalence (%)","DALYs rate per 100000 population"], index=0)
        if selected_m == "Prevalence (%)":
            fig = ff.create_distplot([df_mh[c] for c in df_mh.columns[3:]], group_labels, bin_size=.03,colors=colors)
            fig.update_yaxes(showticklabels=False)
            fig.update_xaxes(showline=True)
            fig.update_layout(height=1000,title_text='Cross-Country Mental Health Disorders Prevalence Histograms',legend=dict(title=dict(text="Disorders")),xaxis=dict(title=dict(text="Prevalence (%)")),yaxis=dict(title=dict(text="Density")))
            st.plotly_chart(fig, use_container_width=True)

        if selected_m == "DALYs rate per 100000 population":
            fig = ff.create_distplot([df_mh_burden[c] for c in df_mh_burden.columns[3:]], group_labels, bin_size=5,colors=colors)
            fig.update_yaxes(showticklabels=False)
            fig.update_xaxes(showline=True)
            fig.update_layout(height=1000,title_text='Cross-Country Mental Health Disorders Burden Histograms',legend=dict(title=dict(text="Disorders")),xaxis=dict(title=dict(text="DALYs rate per 100000 population")),yaxis=dict(title=dict(text="Density")))
            st.plotly_chart(fig, use_container_width=True)

    if selected == "Cross-Country Mental Health Disorders Distribution Plots":
        with st.container(border=True):
            selected_m = st.radio("Select metrics:", ["Prevalence (%)","DALYs rate per 100000 population"], index=0)
        if selected_m == "Prevalence (%)":
            stacked = st.toggle("Stack Distribution Plots", value=False)
            if  stacked == False:
                fig = go.Figure()
                for data_line, color, name in zip([df_mh[c] for c in df_mh.columns[3:]], colors,df_mh.columns[3:]):
                    fig.add_trace(go.Violin(x=data_line, line_color=color,name=f"{name} Disorders"))
                fig.update_traces(orientation='h', side='positive', width=2, points=False)
                fig.update_yaxes(automargin = True)
                fig.update_xaxes(showline=True)
                fig.update_layout(height=1000,xaxis_showgrid=False,xaxis_zeroline=False,title=dict(text='Cross-Country Mental Health Disorders Prevalence Distribution Plots'),legend=dict(title=dict(text="Disorders")),xaxis=dict(title=dict(text="Prevalence (%)")))
                st.plotly_chart(fig, use_container_width=True)  
            if  stacked:
                fig = go.Figure()
                for data_line, color, name in zip([df_mh[c] for c in df_mh.columns[3:]], colors,df_mh.columns[3:]):
                    fig.add_trace(go.Violin(x=data_line, line_color=color,name=f"{name} Disorders"))
                fig.update_traces(orientation='h', side='positive', width=10000000, points=False)
                fig.update_yaxes(color="white")
                fig.update_xaxes(showline=True)
                fig.update_layout(height=500,xaxis_showgrid=False,xaxis_zeroline=False,title=dict(text='Cross-Country Mental Health Disorders Prevalence Distribution Plots'),legend=dict(title=dict(text="Disorders")),xaxis=dict(title=dict(text="Prevalence (%)")))
                st.plotly_chart(fig, use_container_width=True)

        if selected_m == "DALYs rate per 100000 population":
            stacked = st.toggle("Stack Distribution Plots", value=False)
            if  stacked == False:
                fig = go.Figure()
                for data_line, color, name in zip([df_mh_burden[c] for c in df_mh_burden.columns[3:]], colors,df_mh_burden.columns[3:]):
                    fig.add_trace(go.Violin(x=data_line, line_color=color,name=f"{name} Disorders"))
                fig.update_traces(orientation='h', side='positive', width=2, points=False)
                fig.update_yaxes(automargin = True)
                fig.update_xaxes(showline=True)
                fig.update_layout(height=1000,xaxis_showgrid=False,xaxis_zeroline=False,title=dict(text='Cross-Country Mental Health Disorders Burden Distribution Plots'),legend=dict(title=dict(text="Disorders")),xaxis=dict(title=dict(text="DALYs rate per 100000 population")))
                st.plotly_chart(fig, use_container_width=True)

            if  stacked:
                fig = go.Figure()
                for data_line, color, name in zip([df_mh_burden[c] for c in df_mh_burden.columns[3:]], colors,df_mh_burden.columns[3:]):
                    fig.add_trace(go.Violin(x=data_line, line_color=color,name=f"{name} Disorders"))
                fig.update_traces(orientation='h', side='positive', width=10000000, points=False)
                fig.update_yaxes(automargin = True)
                fig.update_xaxes(showline=True)
                fig.update_layout(height=500,xaxis_showgrid=False,xaxis_zeroline=False,title=dict(text='Cross-Country Mental Health Disorders Burden Distribution Plots'),legend=dict(title=dict(text="Disorders")),xaxis=dict(title=dict(text="DALYs rate per 100000 population")))
                st.plotly_chart(fig, use_container_width=True)
    if selected =="Cross-Country Mental Health Disorders Violin Plots":
        with st.container(border=True):
            selected_m = st.radio("Select metrics:", ["Prevalence (%)","DALYs rate per 100000 population"], index=0)
        if selected_m == "Prevalence (%)":
            fig = go.Figure()
            for data_line, color, name in zip([df_mh[c] for c in df_mh.columns[3:]], colors,df_mh.columns[3:]):
                fig.add_trace(go.Violin(y=data_line, line_color=color,name=f"{name} Disorders",box_visible=True))
            fig.update_traces(orientation='v', width=1, points=False)
            fig.update_yaxes(automargin = True)
            fig.update_layout(height=1000,xaxis_showgrid=False,xaxis_zeroline=False,title=dict(text='Cross-Country Mental Health Disorders Prevalence Violin Plots'),legend=dict(title=dict(text="Disorders")),yaxis=dict(title=dict(text="Prevalence (%)")))
            st.plotly_chart(fig, use_container_width=True) 

        if selected_m == "DALYs rate per 100000 population":
            fig = go.Figure()
            for data_line, color, name in zip([df_mh_burden[c] for c in df_mh_burden.columns[3:]], colors,df_mh_burden.columns[3:]):
                fig.add_trace(go.Violin(y=data_line, line_color=color,name=f"{name} Disorders",box_visible=True))
            fig.update_traces(orientation='v', width=1, points=False)
            fig.update_yaxes(automargin = True)
            fig.update_layout(height=1000,xaxis_showgrid=False,xaxis_zeroline=False,title=dict(text='Cross-Country Mental Health Disorders Burden Violin Plots'),legend=dict(title=dict(text="Disorders")),yaxis=dict(title=dict(text="DALYs Rate per 100000 population")))
            st.plotly_chart(fig, use_container_width=True)
    with st.expander("Show interactive graphs explanation ", expanded=True):
        st.markdown('''
                    <center>   
                        
                    **Interactive Graphs 🖱️**    
                        
                    </center>   
                    ''',unsafe_allow_html=True) 
        col1, col2 =st.columns(2,vertical_alignment="center")
        with col1:
            with st.container(border=True,height=220):
                st.markdown('<center>Click on the legend in order to show/hide specific disorders visualizations</center>',unsafe_allow_html=True)   
                x,cent,y=st.columns(3)
                cent.image("./img/legend.png", use_container_width="auto")      
        with col2:
            with st.container(border=True,height=220):   
                st.markdown('<center>You can download plot as png, zoom, pan, reset scale and view on full screen mode by clicking on the tool bar</center>',unsafe_allow_html=True)
                st.image("./img/tool_bar.png", use_container_width="auto")       
    st.markdown("---")

    show_analysis = st.expander("Show Author’s Analysis", expanded=True)
    with show_analysis:
        st.markdown("""
        <h4 style="margin:0 0 8px 0;">✍️ Author’s Analysis</h4>
                        
        <div class="author-box">  

        <p style="margin:0;">         
                    
        These  graphical visualization reveal how each disorder contributes differently to both the frequency and severity of mental health burdens across populations:

        1. **Schizophrenia**:
        The prevalence distribution is highly concentrated near zero, indicating that schizophrenia **affects only a small fraction of the population** globally. However, in the DALYs distribution, the curve extends far to the right, showing a long right-skewed tail. This suggests that, **while rare**, schizophrenia **produces an exceptionally high health burden** in certain regions or populations.

        2. **Depressive Disorders**:
        The **prevalence** curve **is broader and flatter**, extending toward higher percentages, implying that depression is common and widely distributed. In the DALYs plot, depressive disorders show a moderate and relatively symmetric distribution, suggesting a **consistent level of disease burden across populations**. These patterns underline depression’s as a condition with incosistent cross-country frequence, with less a variable severity compared to other disorders.

        3. **Anxiety Disorders**:
        The**prevalence** distribution **closely mirrors** that of **depressive disorders**, though with slightly lower overall density at higher values, indicating similar but somewhat lower population reach. Its **DALYs distribution is narrower and centered around moderate levels**, showing that anxiety contributes steadily but less severely to the total mental health burden. Together, these curves depict anxiety as a **frequent but moderately disabling** disorder, consistent across regions.

        4. **Bipolar Disorder**:
        The **prevalence distribution is narrow and sharply peaked**, concentrated at **low percentages**.Likewise, the **DALYs** distribution remains **tight and near zero**, reflecting **limited global variation and** a **lower** overall **burden**.This shape suggests that **bipolar disorder**, while serious on an individual level, **contributes minimally to** global variability in **mental health outcomes**.

        5. **Eating Disorders**:
        The prevalence curve shows a **modest peak** near the lower range but with slightly **broader dispersion**, implying **regional variability in occurrence**. In contrast, the DALYs distribution features a distinct peak around moderate values, indicating a **notably high individual burden despite low prevalence**. This pattern identifies eating disorders as **high-impact** yet **underrepresented conditions**, but **significant in specific populations**.

        </p>
        <div/>                                                     
            """, unsafe_allow_html=True)
        st.markdown("<br/><br/>", unsafe_allow_html=True)



elif page == "📈 General Cross-Country Progressions":
    section_with_selectbox(
        "📈 General Cross-Country Progressions",
        [
            "Prevalence Trends (1990–2017)",
            "Burden (DALYs) Trends (1990–2017)",
            "Disorder-Specific Progression Comparison",
        ],
    )

elif page == "🗺️ Geographic Regions Based Analysis":
    section_with_selectbox(
        "🗺️ Geographic Regions Based Analysis",
        [
            "Regional Mean Prevalence Comparison",
            "Regional Burden Comparison",
            "Regional Time Series Trends",
        ],
    )

elif page == "💰 Income-Level Based Analysis":
    section_with_selectbox(
        "💰 Income-Level Based Analysis",
        [
            "Prevalence by Income Level",
            "DALYs Rate by Income Level",
            "Temporal Trends by Income Level",
        ],
    )

elif page == "🚩 Outlying Countries Visualization":
    section_with_selectbox(
        "🚩 Outlying Countries Visualization",
        [
            "Top 10 Countries Above Global Mean (by Disorder)",
            "Bottom 10 Countries Below Global Mean (by Disorder)",
            "Z-score Maps or Boxplots",
        ],
    )
#Añadir aqui el dashboard de paises individuales 

elif page == "🔗 Disorders' Metrics Correlations":
    names=['Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar', 'Eating']
    with st.container(border=True):
        selected_m = st.radio("Select metrics:", ["Prevalence","DALYs rate"], index=0)
    if selected_m == "Prevalence":
        data=df_mh_c_y
        title="Prevalence"
    if selected_m == "DALYs rate":
        data=df_mh_burden_c_y
        title="DALYs rate"
    with st.container(border=True):
        selected_v = st.radio("Select graphical visualization:", ["Heatmap","Scatter Plots", "Both"], index=0)
    if selected_v == "Heatmap":
        fig = ff.create_annotated_heatmap(z=data.corr().values,x=names,y=['Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar', 'Eating'],showscale=True,annotation_text=data.corr().values.round(2),xgap=2,ygap=2,visible=True,colorscale=plotly.colors.diverging.Picnic,
                                                zmin=-1,zmax=1,colorbar=dict(title=dict(text="Pearson correlation coefficient",side="top")),font_colors=["black"])
        fig.update_layout(xaxis=dict(side="bottom"),title=dict(text=f"Mental Health Disorders {title} Correlation Heatmap",x=0.45,xanchor='center',font=dict(size=25)))
        fig.layout.height = 800
        st.plotly_chart(fig, use_container_width=True)    
    if selected_v == "Scatter Plots":
        fig=ff.create_scatterplotmatrix(data,diag='histogram', title=f"Mental Health Disorders {title} Correlation Scatter Plots",size=1)
        fig.update_layout(height=1000,title=dict(x=0.5,xanchor='center',font=dict(size=25)))
        fig.update_xaxes(showticklabels=False,title=dict(font=dict(size=15)))
        fig.update_yaxes(showticklabels=False,title=dict(font=dict(size=15)))
        fig.update_traces(marker=dict(line=dict(width=0.1,color='black')))
        st.plotly_chart(fig, use_container_width=True)
    if selected_v == "Both":
        col1, col2 = st.columns([59,39],vertical_alignment="center")
        with col1:
            with st.container(border=True,height=450):
                fig = ff.create_annotated_heatmap(z=data.corr().values,x=names,y=['Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar', 'Eating'],showscale=True,annotation_text=data.corr().values.round(2),xgap=2,ygap=2,visible=True,colorscale=plotly.colors.diverging.Picnic,
                                                zmin=-1,zmax=1,colorbar=dict(title=dict(text="Pearson correlation coefficient",side="top")),font_colors=["black"])
                fig.update_layout(xaxis=dict(side="bottom"),title=dict(text=f"Mental Health Disorders {title} Correlation Heatmap",x=0.42,xanchor='center',font=dict(size=13)))
                fig.layout.height = 410
                fig.layout.width = 560
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            with st.container(border=True,height=450):
                fig=ff.create_scatterplotmatrix(data,diag='histogram', title=f"Mental Health Disorders {title} Correlation Scatter Plots",size=1)
                fig.update_layout(height=410,width=400,title=dict(x=0.5,xanchor='center',font=dict(size=13)))
                fig.update_xaxes(showticklabels=False,title=dict(font=dict(size=7)))
                fig.update_yaxes(showticklabels=False,title=dict(font=dict(size=7)))
                fig.update_traces(marker=dict(line=dict(width=0.1,color='black')))
                st.plotly_chart(fig, use_container_width=True)
    show_analysis = st.expander("Show Author’s Analysis", expanded=True)
    with show_analysis:
        st.markdown("""
        <h4 style="margin:0 0 8px 0;">✍️ Author’s Analysis</h4>           
        <div class="author-box">  
        <p style="margin:0;">         
                      
        
                       
        </p>
        <div/>                                                     
        """, unsafe_allow_html=True)
        st.markdown("<br/><br/>", unsafe_allow_html=True)


elif page == "🌍 Global Choropleth Maps":
    st.header("🌍 Global Choropleth Maps")
    with st.container(border=True):
        selected = st.radio("Select graphical visualization:", ["Global Prevalence Choropleth Maps","Global Burden Choropleth Maps"], index=0)
    if selected == "Global Prevalence Choropleth Maps":
        with st.container(border=True):
            selected_d = st.radio("Select disorder group:", ["Anxiety disorders","Bipolar disorders","Depressive disorders","Eating disorders","Schizophrenia disorders"], index=0)
        st.markdown(" ")
        st.markdown(f"<h5 style='text-align: center;color: gray'>Global {selected_d} Prevalence Choropleth Maps<h5/>",unsafe_allow_html=True)
        selected_key=selected_d.replace(" disorders","")
        geo_data=geo.merge(df_mh_mean,left_on="id",right_on="code",how="inner")
        colormap = cm.linear.YlOrRd_09.scale(geo_data[selected_key].min(), geo_data[selected_key].max()).to_step(7)
        m = folium.Map(location=[0,0],zoom_start=2,width=1030,height=1000,control_scale=True)
        folium.TileLayer('CartoDB positron',name="Light Map",control=False,attr = "Luna Pérez Troncoso").add_to(m)
        colormap.caption = f"{selected_key} Disorders Prevalence %"
        style_function = lambda x: {"weight":0.1,'color':'black','fillColor':colormap(x['properties'][selected_key]), 'fillOpacity':0.9}
        highlight_function = lambda x: {'fillColor': '#000000', 'color': '#FFFFFF', 'fillOpacity': 0.5, 'weight': 0.5}
        NIL=folium.features.GeoJson(geo_data,style_function=style_function,control=False,highlight_function=highlight_function,tooltip=folium.features.GeoJsonTooltip(fields=['name',selected_key],aliases=['Country',f'{selected_key} Disorders Prevalence %'],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"),sticky=True))
        colormap.add_to(m)
        m.add_child(NIL)
        sf.st_folium(m,width=1030, height=1000)

        st.markdown("<br/><br/>", unsafe_allow_html=True)
        st.markdown("---")

        show_dataframe1 = st.expander("Show Fold Change Pertentage DataFrame", expanded=False)       
        with show_dataframe1:
            st.image("./img/bar2.png",use_container_width= "always")
            cmap = mpl.colormaps['bwr']
            st.dataframe(pd.DataFrame(100*df_mh_mean2.values/df_mh_world_mean.values,columns=df_mh_mean2.columns,index=df_mh_mean2.index).style.background_gradient(cmap=cmap, vmin=50,vmax=150),use_container_width=True)
            st.markdown("<br/><br/>", unsafe_allow_html=True)
        st.markdown("---")

        show_analysis = st.expander("Show Author’s Analysis", expanded=True)
        with show_analysis:
            st.markdown("""
            <h4 style="margin:0 0 8px 0;">✍️ Author’s Analysis</h4>
                        
            <div class="author-box">  

            <p style="margin:0;">         
                    
            **Schizophrenia**
            - **Highest deviations:** The **United States**, **New Zealand**, **Australia**, and several **Western European nations** (Netherlands, Ireland) display **substantially higher prevalence** than the global mean.   
            - **Lowest deviations:** **Sub-Saharan African nations** such as **Somalia, Malawi, Mozambique, Burundi**, and the **Democratic Republic of Congo** are positioned **far below the world average**.  
            - **Interpretation:** The pattern suggests that **schizophrenia is more frequently diagnosed in high-income contexts**, likely due to **better detection systems**, while **underdiagnosis and limited psychiatric resources** explain the low apparent prevalence in poorer regions.

            **Depressive Disorders**
            - **Highest deviations:** The **most extreme values** are found in **African and Middle Eastern countries** like **Uganda, Palestine, Greenland, and the Central African Republic**.
            - **Lowest deviations:** On the other end, **East Asian countries** (notably **Japan, South Korea, Singapore, and Myanmar**) display the **lowest fold changes**,  **60% to 45%**.  
            - **Interpretation:** These results point to a **dual cultural and socioeconomic influence**,**high measured depression** in low-resource regions may reflect **social instability, conflict, and limited support systems**, while **East Asian underrepresentation** could stem from **underreporting due to stigma and cultural norms** that discourage seeking mental health care.

            **Anxiety Disorders**
            - **Highest deviations:** **Portugal, Brazil, New Zealand, and several Northern European nations** (Norway, Ireland, Switzerland, Netherlands) show **significantly elevated anxiety prevalence**.  
            - **Lowest deviations:** Conversely, **Central and East Asian countries**, including **Uzbekistan, Kyrgyzstan, Mongolia, and Vietnam**, fall **well below the global mean**.  
            - **Interpretation:** Anxiety appears **most pronounced in Western societies**, where **lifestyle stressors and cultural openness to diagnosis** contribute to higher measured rates, while **lower values in Asia** likely reflect a **combination of cultural minimization and limited diagnostic exposure**.

            **Bipolar Disorders**
            - **Highest deviations:** The **Oceania and Western Hemisphere** dominate the top ranks, **New Zealand, Australia, Brazil, the UK, and several South American countries** exhibit **much higher prevalence** than the global average.  
            - **Lowest deviations:** **East Asian and Pacific island countries** (e.g., **China, North Korea, Taiwan, Papua New Guinea, Micronesia**) display **substantially lower levels**.  
            - **Interpretation:** This bipolar pattern suggests **strong regional clustering**, where **diagnostic sophistication and awareness** in wealthier regions contrast with **significant underrecognition** in developing nations.

            **Eating Disorders**
            - **Highest deviations:** The **most striking differences globally**, with **Australia, Monaco, New Zealand, Spain, and Italy** showing **massive positive deviations**.  
            - **Lowest deviations:** **Sub-Saharan and Southeast Asian countries** such as **Somalia, Ethiopia, Myanmar, Cambodia**, and **Mozambique** appear **far below the mean**.  
            - **Interpretation:** Eating disorders are **heavily concentrated in high-income Western contexts**, strongly tied to **cultural ideals of thinness and social comparison**, while **largely absent or underdetected** in regions with different beauty norms or food insecurity.

            In conclusion, the results reveal a **clear geographical and socioeconomic polarization**, where **high-income nations tend to exceed global means**, while **low-income and lower-middle-income countries fall well below them**. **Wealthier nations show higher prevalence and greater diagnostic visibility**, **poorer nations remain underrepresented in clinical data** — masking potentially significant unmet needs. Breaking this trend, in **depressive disorders**, some **African and conflict-affected regions** surpass global averages, reflecting **psychosocial stressors** and **humanitarian crises** rather than diagnostic bias. The results thus reinforce that **mental health prevalence patterns are shaped not only by biology, but by culture, reporting practices, and structural disparities in healthcare access**.

            </p>
            <div/>                                                     
            """, unsafe_allow_html=True)
            st.markdown("<br/><br/>", unsafe_allow_html=True)


    if selected == "Global Burden Choropleth Maps":
        with st.container(border=True):
            selected_d = st.radio("Select disorder group:", ["Anxiety disorders","Bipolar disorders","Depressive disorders","Eating disorders","Schizophrenia disorders"], index=0)
        st.markdown(" ")
        st.markdown(f"<h5 style='text-align: center;color: gray'>Global {selected_d} DALYs Rate Choropleth Maps<h5/>",unsafe_allow_html=True)
        selected_key=selected_d.replace(" disorders","")
        geo_data=geo.merge(df_mh_burden_mean,left_on="id",right_on="code",how="inner")
        colormap = cm.linear.YlOrRd_09.scale(geo_data[selected_key].min(), geo_data[selected_key].max()).to_step(7)
        m = folium.Map(location=[0,0],zoom_start=2,width=1030,height=1000,control_scale=True)
        folium.TileLayer('CartoDB positron',name="Light Map",control=False,attr = "Luna Pérez Troncoso").add_to(m)
        colormap.caption = f"{selected_key} Disorders DALYs Rate per 100000 population"
        style_function = lambda x: {"weight":0.1,'color':'black','fillColor':colormap(x['properties'][selected_key]), 'fillOpacity':0.9}
        highlight_function = lambda x: {'fillColor': '#000000', 'color': '#FFFFFF', 'fillOpacity': 0.5, 'weight': 0.5}
        NIL=folium.features.GeoJson(geo_data,style_function=style_function,control=False,highlight_function=highlight_function,tooltip=folium.features.GeoJsonTooltip(fields=['name',selected_key],aliases=['Country',f'{selected_key} Disorders DALYs Rate per 100000 population'],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"),sticky=True))
        colormap.add_to(m)
        m.add_child(NIL)
        sf.st_folium(m,width=1030, height=1000)
        st.markdown("<br/><br/>", unsafe_allow_html=True)
        st.markdown("---")

        show_dataframe2 = st.expander("Show Fold Change Percentage DataFrame", expanded=False)
        with show_dataframe2:
            cmap = mpl.colormaps['bwr']
            st.dataframe(pd.DataFrame(100*df_mh_burden_mean2.values/df_mh_burden_world_mean.values,columns=df_mh_burden_mean2.columns,index=df_mh_burden_mean2.index).style.background_gradient(cmap=cmap, vmin=50,vmax=150),use_container_width=True)
        st.markdown("---")

        show_analysis = st.expander("Show Author’s Analysis", expanded=True)
        with show_analysis:
            st.markdown("""
            <h4 style="margin:0 0 8px 0;">✍️ Author’s Analysis</h4>
                          
            <div class="author-box">  

            <p style="margin:0;">         
                    
            **Schizophrenia**
            - **Highest-burden countries:** **Uganda, Palestine, Greenland, Central African Republic, Congo, Angola, Equatorial Guinea, Gabon, Gambia, Greece**.  
            - **Lowest-burden countries:** Sub-Saharan African and several East Asian nations.  
            - **Interpretation:** Elevated DALY rates in certain African and conflict-affected regions reflect **high disease burden**, whereas lower rates in other low-income countries may partly result from **limited diagnostic capacity**, suggesting potential underestimation.

            **Depressive Disorders**
            - **Highest-burden countries:** **United States, New Zealand, Australia, Netherlands, Greenland, Ireland, Guam, Northern Mariana Islands, Singapore, Vietnam**.  
            - **Lowest-burden countries:** East Asian nations (Japan, South Korea, Myanmar) and some low-income regions.  
            - **Interpretation:** Burden is elevated in high-income countries due to **higher reporting and treatment access**, whereas cultural norms and healthcare limitations may contribute to lower apparent DALYs in other regions.

            **Anxiety Disorders**
            - **Highest-burden countries:** **New Zealand, Northern Ireland, Australia, England, Brazil, Paraguay, UK, Israel, Argentina, Chile**.  
            - **Lowest-burden countries:** Central and East Asian nations, including Vietnam, Kyrgyzstan, Uzbekistan.  
            - **Interpretation:** DALY burden for anxiety is concentrated in Western countries, likely reflecting both **lifestyle stressors and healthcare detection**, while low-burden regions may experience **underdiagnosis**.

            **Bipolar Disorders**
            - **Highest-burden countries:** **Australia, Monaco, New Zealand, Spain, Italy, Luxembourg, Austria, San Marino, Andorra, Switzerland**.  
            - **Lowest-burden countries:** East Asian and Pacific Island nations.  
            - **Interpretation:** High DALY rates indicate **substantial disease burden** in wealthier countries with robust diagnostic systems, whereas low values in developing regions may reflect **underrecognition and reporting gaps**.

            **Eating Disorders**
            - **Highest-burden countries:** **Portugal, Brazil, New Zealand, Iran, Northern Ireland, Norway, Netherlands, Ireland, Switzerland, Cyprus**.  
            - **Lowest-burden countries:** Sub-Saharan and Southeast Asian countries.  
            - **Interpretation:** Burden is heavily concentrated in high-income contexts, driven by **cultural and lifestyle factors**, while low-burden regions may face **diagnostic gaps or different cultural drivers of disease**.

            Wealthier nations show **consistently higher DALY rates** across multiple mental health disorders. Lower-income countries often appear underrepresented, although some conflict-affected regions (e.g., certain African countries) show extreme burden. **Bipolar and anxiety disorders exhibit DALY rates 6 times above the global mean**, illustrating concentrated disease burden **in specific countries**. The combined analysis of countries with **highest and lowest DALY rates** demonstrates that mental health burden is **shaped by socioeconomic, cultural, and healthcare factors**, not only biological risk. These results highlight the need for **globally targeted mental health strategies** that address both **underdiagnosis in low-resource regions** and **high disease burden in wealthier countries**.

            </p>
            <div/>                                                     
            """, unsafe_allow_html=True)

        st.markdown("<br/><br/>", unsafe_allow_html=True)


elif page == "🕰️ Comparison of Choropleth Maps by Year":
    st.header("🕰️ Comparison of Choropleth Maps by Year")
    with st.container(border=True):
        selected = st.radio("Select graphical visualization:", ["Global Prevalence Comparison Choropleth Maps","Global Burden Comparison Choropleth Maps"], index=0)
    if selected == "Global Prevalence Comparison Choropleth Maps":
        with st.container(border=True):
            selected_d = st.radio("Select disorder group:", ["Anxiety disorders","Bipolar disorders","Depressive disorders","Eating disorders","Schizophrenia disorders"], index=0)
        years = st.slider("Select the years of the comparison",1990,2019,(1990,2019),1)
        geo_data_1990=geo.merge(df_mh_burden[df_mh_burden.year==years[0]],left_on="id",right_on="code",how="inner")
        geo_data_2019=geo.merge(df_mh_burden[df_mh_burden.year==years[1]],left_on="id",right_on="code",how="inner")
        df_mh_1990=df_mh[df_mh.year==years[0]].drop(["year","code"],axis=1).set_index("country")
        df_mh_2019=df_mh[df_mh.year==years[1]].drop(["year","code"],axis=1).set_index("country")
        st.markdown(" ")
        st.markdown(f"<h5 style='text-align: center;color: gray'>Global {selected_d} Prevalence Choropleth Maps<h5/>",unsafe_allow_html=True)
        selected_key=selected_d.replace(" disorders","")
        colormap = cm.linear.YlOrRd_09.scale(geo_data_1990[selected_key].min(), geo_data_2019[selected_key].max()).to_step(7)
        m = plugins.DualMap(location=[0,0],zoom_start=1,control_scale=True)
        folium.TileLayer('CartoDB positron',name="Light Map",control=False,attr = "Luna Pérez Troncoso").add_to(m)
        colormap.caption = f"{selected_key} Disorders Prevalence %"
        style_function = lambda x: {"weight":0.1,'color':'black','fillColor':colormap(x['properties'][selected_key]), 'fillOpacity':0.9}
        highlight_function = lambda x: {'fillColor': '#000000', 'color': '#FFFFFF', 'fillOpacity': 0.5, 'weight': 0.5}
        NIL_1990=folium.features.GeoJson(geo_data_1990,style_function=style_function,control=False,highlight_function=highlight_function,tooltip=folium.features.GeoJsonTooltip(fields=['name',selected_key],aliases=['Country',f'{selected_key} Disorders Prevalence %'],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"),sticky=True))
        NIL_2019=folium.features.GeoJson(geo_data_2019,style_function=style_function,control=False,highlight_function=highlight_function,tooltip=folium.features.GeoJsonTooltip(fields=['name',selected_key],aliases=['Country',f'{selected_key} Disorders Prevalence %'],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"),sticky=True))
        colormap.add_to(m.m2)
        m.m1.add_child(NIL_1990)
        m.m2.add_child(NIL_2019)
        sf.folium_static(m,width=1030, height=500)
        st.markdown(f"<h5 style='text-align: center;color: black'> {str(years[0])} &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; {str(years[1])} <h5/>",unsafe_allow_html=True)
        st.markdown("---")

        show_dataframe1 = st.expander("Show Fold Change Pertentage DataFrame", expanded=False)       
        with show_dataframe1:
            st.image("./img/bar2.png",use_container_width= "always")
            cmap = mpl.colormaps['bwr']
            st.dataframe((df_mh_2019/df_mh_1990*100).style.background_gradient(cmap=cmap, vmin=50,vmax=150),use_container_width=True)
            st.markdown("<br/><br/>", unsafe_allow_html=True)
        st.markdown("---")
        show_analysis=st.expander("Show Author’s Analysis", expanded=True)
        with show_analysis:
            st.markdown("""
            <h4 style="margin:0 0 8px 0;">✍️ Author’s Analysis</h4>
                        
            <div class="author-box">  

            <p style="margin:0;">         
                    
            Fold-change (FC) values describe the relative change in disorder prevalence between **1990 and 2019**:
            - **FC = 100** → no net change.
            - **FC > 100** → increased prevalence.
            - **FC < 100** → decreased prevalence.

            Across all disorders and 190+ countries, FC values generally cluster between **95 and 110**, indicating **moderate but widespread shifts** rather than explosive growth or collapse. However, certain disorders and countries show **systematic trends**, suggesting changing sociocultural, diagnostic, and epidemiological dynamics. **Stable biological disorders** (schizophrenia, bipolar) remained constant. **Culturally and psychosocially sensitive disorders** (anxiety, depression, eating) have **expanded considerably**, especially in **urbanized, high-income, or rapidly modernizing societies**. Geographic differences reveal that **Western lifestyle patterns and diagnostic capacity** amplify recorded prevalence, while **limited surveillance in low-income regions** may obscure true changes.

            **Cross-Disorder Global Trends**

            | Disorder | Global Pattern | Key Insights |
            |-----------|----------------|---------------|
            | **Schizophrenia** | Stable (mostly 97–103) | Prevalence remained almost unchanged globally. Small growth in parts of SE Asia and Eastern Europe (e.g., Cambodia, Myanmar), slight declines in Western Europe. Reflects biological constancy and limited diagnostic drift. |
            | **Depressive Disorders** | Highly variable (80–115) | Strong polarization: sharp decreases (<90) in Western & Northern Europe (Austria, France, Finland), but **substantial increases** (>110) in parts of Latin America (Mexico, Uruguay) and Asia (Malaysia). Suggests rising detection and societal stress in developing high-income contexts. |
            | **Anxiety Disorders** | Systematic increase (100–110, occasionally >120) | Most consistent global rise. Largest surges in **Americas, Middle East, and South-East Asia** (Brazil, Turkey, Nepal). Only limited declines (Japan, Norway). Indicates intensification of stress-related conditions linked to urbanization, digital exposure, and economic transitions. |
            | **Bipolar Disorders** | Near-neutral (≈100) | Minimal variation globally. Small positive deviations in Australia, Uruguay, Ireland; negligible change elsewhere. Likely reflects diagnostic stability and long-term chronic nature of the condition. |
            | **Eating Disorders** | Strong upward trend, highly heterogeneous | The **most dynamic disorder**: FC values frequently exceed **130–170** (e.g., China, Myanmar, Laos, Vietnam, Australia). Indicates dramatic expansion of prevalence, tied to **Westernization, media exposure, and shifting beauty ideals**. Isolated declines (<95) in poorer African and Pacific regions (e.g., Libya, Somalia). |

            **Regional and Cultural Patterns**

            - **Europe & North America:**  
                - Mixed evolution. Depressive and anxiety disorders largely increased in Southern and Eastern Europe (Portugal, Poland), while Northern Europe often shows slight declines in depression but **stable or rising anxiety**. Eating disorders exhibit the **strongest growth** in high-income Western countries, consistent with cultural and societal pressures.

            - **Asia-Pacific:**  
                - Marked **surge in eating and anxiety disorders**, especially in East and South-East Asia (China +72%, Vietnam +54%, Thailand +40%). Rapid modernization, digital globalization, and changing body image norms contribute to this trend. Schizophrenia and bipolar remain mostly unchanged.

            - **Middle East & North Africa:**  
                - Moderate increases in anxiety and eating disorders (Iran, Egypt, Saudi Arabia). Depression trends are mixed, with some declines (Libya) likely linked to socio-political instability and under-reporting.

            - **Sub-Saharan Africa:**  
                - Slight overall increases in anxiety and schizophrenia, but **limited diagnostic growth**. Some countries (Uganda, Ghana, Tanzania) show small positive FCs, while others (Burundi, Somalia) exhibit declines, possibly due to limited health surveillance and diagnostic access.

            - **Latin America:**  
                - Clear upward trajectory for anxiety and depression in several nations (Mexico, Chile, Uruguay, Brazil). Cultural shifts, economic instability, and greater recognition of mental health are probable drivers. Eating disorders also rising sharply.

            - **High-Income Island States (Australia, New Zealand, Malta):**  
                - Exhibit some of the **strongest positive fold changes** in eating disorders (+130–145), reflecting both higher awareness and genuine prevalence increase.  
            
            **Notable Outliers**

            - **Equatorial Guinea (Eating FC ≈ 246)** – extreme outlier; may reflect data artefacts or emerging high-income consumption patterns.
            - **China (Eating FC ≈ 172)** – one of the strongest true increases, aligning with cultural westernization.
            - **Singapore (Depression FC ≈ 71)** – significant decline; possible under-detection or public-health progress.
            - **Norway (Depression FC ≈ 109)** and **United States (Depression FC ≈ 115)** – rising depression levels despite high income, supporting the stress–diagnosis hypothesis.
                                    
            </p>
            <div/>                                                     
            """, unsafe_allow_html=True)
            st.markdown("<br/><br/>", unsafe_allow_html=True)


    if selected == "Global Burden Comparison Choropleth Maps":
        with st.container(border=True):
            selected_d = st.radio("Select disorder group:", ["Anxiety disorders","Bipolar disorders","Depressive disorders","Eating disorders","Schizophrenia disorders"], index=0)
        years = st.slider("Select the years of the comparison",1990,2019,(1990,2019),1)
        geo_data_1990b=geo.merge(df_mh_burden[df_mh_burden.year==years[0]],left_on="id",right_on="code",how="inner")
        geo_data_2019b=geo.merge(df_mh_burden[df_mh_burden.year==years[1]],left_on="id",right_on="code",how="inner")
        df_mh_burden_1990=df_mh_burden[df_mh_burden.year==years[0]].drop(["year","code"],axis=1).set_index("country")
        df_mh_burden_2019=df_mh_burden[df_mh_burden.year==years[1]].drop(["year","code"],axis=1).set_index("country")
        st.markdown(" ")
        st.markdown(f"<h5 style='text-align: center;color: gray'>Global {selected_d} DALYs Rate Choropleth Maps<h5/>",unsafe_allow_html=True)
        selected_key=selected_d.replace(" disorders","")
        colormap = cm.linear.YlOrRd_09.scale(geo_data_1990b[selected_key].min(), geo_data_2019b[selected_key].max()).to_step(7)
        m = plugins.DualMap(location=[0,0],zoom_start=1,control_scale=True)
        folium.TileLayer('CartoDB positron',name="Light Map",control=False,attr = "Luna Pérez Troncoso").add_to(m)
        colormap.caption = f"{selected_key} Disorders DALYs Rate per 100000 population"
        style_function = lambda x: {"weight":0.1,'color':'black','fillColor':colormap(x['properties'][selected_key]), 'fillOpacity':0.9}
        highlight_function = lambda x: {'fillColor': '#000000', 'color': '#FFFFFF', 'fillOpacity': 0.5, 'weight': 0.5}
        NIL_1990=folium.features.GeoJson(geo_data_1990b,style_function=style_function,control=False,highlight_function=highlight_function,tooltip=folium.features.GeoJsonTooltip(fields=['name',selected_key],aliases=['Country',f'{selected_key} Disorders DALYs Rate per 100000 Population'],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"),sticky=True))
        NIL_2019=folium.features.GeoJson(geo_data_2019b,style_function=style_function,control=False,highlight_function=highlight_function,tooltip=folium.features.GeoJsonTooltip(fields=['name',selected_key],aliases=['Country',f'{selected_key} Disorders DALYs Rate per 100000 Population'],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"),sticky=True))
        colormap.add_to(m.m2)
        m.m1.add_child(NIL_1990)
        m.m2.add_child(NIL_2019)
        sf.folium_static(m,width=1030, height=500)
        st.markdown(f"<h5 style='text-align: center;color: black'> {str(years[0])} &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; {str(years[1])} <h5/>",unsafe_allow_html=True)
        st.markdown("---")

        show_dataframe2 = st.expander("Show Fold Change Percentage DataFrame", expanded=True)
        with show_dataframe2:
            cmap = mpl.colormaps['bwr']
            st.dataframe((df_mh_burden_2019/df_mh_burden_1990*100).style.background_gradient(cmap=cmap, vmin=50,vmax=150),use_container_width=True)
        st.markdown("---")

        show_analysis = st.expander("Show Author’s Analysis", expanded=True)
        with show_analysis:
            st.markdown("""
            <h4 style="margin:0 0 8px 0;">✍️ Author’s Analysis</h4>
                          
            <div class="author-box">  

            <p style="margin:0;">         
                    
            Overall, the distribution of DALYs fold-change values shows **limited global variation** (typically between 95 and 110), but **directionally divergent trends** emerge across disorders and regions. These differences reveal how **medical progress, health system capacity, and cultural factors** shape the actual lived impact of mental illness beyond prevalence alone. **Chronic and biologically rooted disorders** (schizophrenia, bipolar) maintained or increased their burden in developing regions, highlighting care inequities. **Culturally sensitive disorders** (depression, anxiety, eating) show rising burden in **high-income and rapidly modernizing countries**, where **social pressures and digital environments magnify psychological strain**. In summery, developed countries' **individuals **live longer but with persistent functional impairment**.

            **Cross-Disorder Global Trends**

            | Disorder | Global Pattern | Key Insights |
            |-----------|----------------|---------------|
            | **Schizophrenia** | Mild decrease globally (≈95–100), but increases in South & East Asia (India, Malaysia, South Korea, Vietnam). | Stable prevalence but variable outcomes: countries with limited long-term treatment (South/Southeast Asia, parts of Eastern Europe) show rising DALY rates, indicating persistent disability despite possible diagnostic progress. |
            | **Depressive Disorders** | Broad stability (≈98–104) with moderate increases in low- and middle-income nations. | Reflects partial improvements in management in high-income regions (Europe, North America) but growing severity and chronicity elsewhere. Suggests that awareness increased faster than effective care. |
            | **Anxiety Disorders** | Almost universally stable (≈99–101). | Although prevalence increased, the **burden per capita remained constant**, suggesting that anxiety disorders are **better managed or less disabling** despite spreading more widely. High-income countries show DALY plateaus, pointing to therapeutic containment. |
            | **Bipolar Disorder** | Strong and consistent increases (110–160, peaking at >170 in China, Myanmar, Vietnam). | Indicates a **rising chronic disability load**, even where prevalence barely changed. Suggests that individuals live longer with bipolar disorder but with sustained impairment, reflecting treatment-resistant or recurrent forms. |
            | **Eating Disorders** | Mild but widespread increase (≈102–108, locally >120 in some Asian and American nations). | The expansion in DALYs is less pronounced than the rise in prevalence, implying some improvement in case management. However, countries such as the UK, Spain, and Turkey show marked growth (>105), revealing persistent disability linked to cultural and social pressures. |

            **Regional and Cultural Patterns**

            - **Europe & North America:**  
                - Burden **declines slightly** for schizophrenia and depression (e.g., France, Finland, UK).  
                - **Increases for bipolar and eating disorders**, especially in Southern and Western Europe (Spain, Portugal, Italy, UK).  
                - Suggests that while medical treatment reduces mortality and some disability, **sociocultural pressures sustain functional burden** for affective and eating disorders.  
                - The **United States and Uruguay** stand out with high DALYs increases in multiple disorders (schizophrenia +22%, anxiety +5%, eating +8%), indicating a persistent mental health strain despite advanced healthcare access.

            - **Asia-Pacific:**  
                - Strong upward trends in **bipolar and schizophrenia DALYs** (India +50%, China +71%, Thailand +40%, Vietnam +54%).  
                - **Eating disorder burden rises** moderately in high-growth economies (Malaysia, Laos, Thailand).  
                - Reflects the **psychosocial cost of rapid modernization**, cultural transition, and unequal access to long-term psychiatric care.

            - **Middle East & North Africa:**  
                - Moderate increases in burden for bipolar and eating disorders (Iran, Egypt, Saudi Arabia).  
                - Slight reductions in depression and anxiety suggest growing treatment capacity in wealthier states.  
                - In contrast, **Libya** and **Syria** show mixed signals—declines in prevalence but persistent or worsening DALY rates—hinting at underreporting and disrupted healthcare access.

            - **Sub-Saharan Africa:**  
                - Generally **stable or slightly increased DALYs** (≈101–105 across disorders).  
                - Stronger rises for **bipolar** and **eating disorders** in Uganda, Tanzania, and Mozambique.  
                - Reflects emerging recognition and improved survival rather than an actual rise in disease frequency.

            - **Latin America:**  
                - **Increasing DALYs** for affective and bipolar disorders (Mexico +20%, Brazil +15%).  
                - Cultural modernization and health system strain drive worsening disability profiles, even as prevalence stabilizes.  
                - **Eating disorders** show particularly strong growth in Brazil and Mexico, reinforcing the sociocultural link between economic development and mental health burden.

            - **High-Income Island States (Australia, New Zealand, Iceland):**  
                - Bipolar disorder DALYs rose steeply (Australia +43%), suggesting that **chronic management improvements extend lifespan but not full recovery**.  
                - Slight gains in eating disorder DALYs mirror global cultural exposure and urban pressures.

            **Notable Outliers**

            - **United States (+22.7% in schizophrenia, +8.4% in eating disorders)** – suggests increasing functional impairment despite stable prevalence.  
            - **Myanmar and Vietnam (>160% DALYs increase for bipolar)** – extreme outliers driven by regional health inequities and limited psychiatric coverage.  
            - **Spain and Turkey** – strong increases across several disorders, reflecting cultural convergence and diagnostic intensification.  
            - **Singapore** – marked decline in schizophrenia DALYs (−32%), possibly due to early intervention programs and healthcare modernization.  

            </p>
            <div/>                                                     
            """,unsafe_allow_html=True)

        st.markdown("<br/><br/>", unsafe_allow_html=True)

elif page=="✍️ Conclusions":
    st.markdown('''
    ### ✍️ Conclusions on Global Mental Health Data:

    The combined analyses of prevalence trends, DALYs rates, and top/bottom country comparisons reveal **complex, disorder-specific, income-sensitive, and geographically influenced patterns** in global mental health:  

    **General Patterns:**  
    - All five major mental health disorders show **statistically significant temporal changes** in both prevalence and DALYs rates ($p < 10^{-4}$), demonstrating that mental health challenges are **dynamic and evolving worldwide**. Countries with the **highest DALYs rates** exhibit burdens **150–168% above global averages**, while those with the lowest remain **68–80% below**, highlighting **global disparities in burden, detection, and treatment access**.

    **Income-Level Associations:**  
    - Except for schizophrenia, **high-income countries** consistently display **higher prevalence and greater disability burden**, indicating that mental health challenges are **more impactful and visible** in developed contexts. Some low-income regions may **underreport or underdiagnose** mental disorders, masking true disability.

    **Geography and Cultural Influences:**  
    - **Region of the Americas and Europe** consistently show the **highest prevalence and DALY rates across most disorders**, reflecting **greater detection, lifestyle-related stressors, and years lived with disability**.  
    - **Africa and South-East Asia/Asia** report **lower prevalence**, but **higher DALYs in some disorders** (e.g., schizophrenia), suggesting **underdiagnosis combined with limited treatment access**.  
    - Hypothesis testing confirms these geographic differences are **highly significant** ($p < 10^{-20}$), indicating **real variation beyond random fluctuation**.  
    - Cultural factors, such as **societal attitudes toward mental health, stigma, lifestyle norms, and healthcare-seeking behavior**, likely modulate both **reported prevalence and disability**.

    **Disorder-Specific Insights:**  
    - **Schizophrenia:** Burden is greatest in low-income countries despite uniform prevalence (~0.2–0.3%), reflecting **limited access to sustained care and medication**.  
    - **Anxiety Disorders:** Highest in high-income nations (~5% prevalence, DALYs double low-income regions), linked to **stress-intensive lifestyles and complete diagnostic coverage**.  
    - **Bipolar Disorder:** Stable prevalence, but DALYs rise sharply in wealthier nations (3–4× compared to low-income regions), reflecting **chronicity, long-term disability, and detection of milder cases**.  
    - **Depressive Disorders:** High-income countries lead both in prevalence and DALYs (~4.5–5% prevalence, >210 per 100,000 DALYs), reinforcing depression as a **major contributor to global mental health burden**.  
    - **Eating Disorders:** Concentrated in high-income nations (~0.4% prevalence, >470 DALYs per 100,000), reflecting **cultural pressures, beauty standards, and lifestyle expectations**.

    **Top vs. Bottom Country Patterns:**  
    - Countries with the **highest DALYs rates** (e.g., Palestine, Greenland, Australia) show **dramatically higher disease burden**, reflecting **true epidemiological differences or enhanced reporting**.  
    - Countries with **lowest DALYs rates** (e.g., Vietnam, Myanmar, South Korea) have **substantially lower measured burden**, likely reflecting **underdiagnosis, healthcare limitations, or cultural barriers to detection**.

    **Key Takeaways:**  
    - Mental health burden **does not simply decrease with income**; high-income contexts may amplify both **prevalence and functional impact**, especially for anxiety, depression, and eating disorders.  
    - Schizophrenia remains an exception: **economic disadvantage correlates with greater disability**, underscoring the **critical need for global access to psychiatric care**.  
    - Geographic and cultural context shapes both **detection and disability**, highlighting the need for **locally tailored interventions** alongside universal strategies.  
    - These patterns underscore the importance of **targeted, disorder-specific, income- and region-aware public health policies** to reduce the global burden of mental disorders.

    In conclusion, global mental health trends are **heterogeneous, income-sensitive, geographically influenced, and disorder-specific**, requiring **strategic, context-aware public health policies** to address both **rising burdens in high-income nations** and **treatment gaps in low-income or underdiagnosed regions**.
    ''')
else:
    st.header("🙋🏻‍♀️ About the Author")
    st.markdown('''
    I’m **Luna Pérez Troncoso**, a **Data Scientist** with a strong foundation in **Artificial Intelligence, data analytics, and computational modeling**, originally shaped through my background in **science and research**. Over time, I’ve transitioned my **analytical and experimental mindset** into the tech field, where I design **data-driven solutions** that **transform complex information into actionable insights**.
         
    I have experience working across the **full data pipeline**, from data acquisition, cleaning, and exploration to building, validating, and deploying predictive models. My focus is on **leveraging machine learning and statistical methods to uncover patterns, optimize processes, and support strategic decisions**.
           
    What defines my approach is a balance between **technical precision and creativity**. I’m passionate about **connecting raw data with real-world** impact, collaborating with **cross-functional teams**, and communicating insights in a way that drives innovation.
    ''')
    col1, col2= st.columns(2,gap="large",vertical_alignment="center")
    with col1:
        with st.container(border=True):
            col1_1, col1_2 = st.columns(2,vertical_alignment="center")
            with col1_1:
                st.image("./img/linkedin.jpg",use_container_width=True)
            with col1_2:
                st.markdown('''
                <center>
                                
                Visit my [linkedIn profile](https://www.linkedin.com/in/luna-p%C3%A9rez-troncoso-0ab21929b/)
                            
                </center>
                ''',unsafe_allow_html=True)
    with col2:
        with st.container(border=True):
            col2_1, col2_2 = st.columns(2,vertical_alignment="center")
            with col2_1:
                st.image("./img/github.png",use_container_width=True)
            with col2_2:
                st.markdown('''
                <center>
                                
                Explore my projects on my [Github profile](https://github.com/LunaPerezT)
                            
                </center>
                ''',unsafe_allow_html=True)


# ---------- FOOTER ----------
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 1em;'>© 2025 Global Mental Health EDA — Luna Pérez Troncoso </p>",unsafe_allow_html=True)