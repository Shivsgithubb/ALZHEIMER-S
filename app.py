from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

app = Flask(__name__)
app.secret_key = 'supersecretmre'

df = pd.read_csv('alzheimers_prediction_dataset.csv')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login')
def login():
    return render_template('login.html')

#graphs functions
def age_distribution():
    bins = [0, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ['<30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90+']
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    # Calculate diagnosis rate by age group
    age_analysis = df.groupby('Age Group')["Alzheimer’s Diagnosis"].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
    age_analysis.columns = ['Age Group', 'Diagnosis Rate (%)']
    # Create Plotly bar chart
    fig = px.bar(age_analysis, 
                x='Age Group', 
                y='Diagnosis Rate (%)',
                title='Alzheimer\'s Diagnosis Rate by Age Group',
                labels={'Age Group': 'Age Range', 'Diagnosis Rate (%)': 'Diagnosis Rate (%)'},
                color='Diagnosis Rate (%)',
                color_continuous_scale='Viridis')
    fig.update_layout(
        title_x=0.5,
        plot_bgcolor='white',
        showlegend=False,
        width=800,
        height=500
    )
    fig.update_traces(
        texttemplate='%{y:.1f}%',
        textposition='outside'
    )
    fig.update_xaxes(title_font=dict(size=12), tickfont=dict(size=10))
    fig.update_yaxes(title_font=dict(size=12), tickfont=dict(size=10))
    graph1_html = pio.to_html(fig, full_html=False)
    return graph1_html

def gender_distribution():
    # Calculate gender distribution for diagnosed and non-diagnosed cases
    gender_diagnosis = pd.crosstab(df['Gender'], df["Alzheimer’s Diagnosis"])
    # Create two subplots
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('Gender Distribution - Diagnosed', 'Gender Distribution - Not Diagnosed'),
                        specs=[[{'type':'domain'}, {'type':'domain'}]])
    # Add pie charts
    fig.add_trace(go.Pie(labels=gender_diagnosis.index, 
                        values=gender_diagnosis['Yes'],
                        name="Diagnosed",
                        marker_colors=['#2ecc71', '#3498db']),
                1, 1)
    fig.add_trace(go.Pie(labels=gender_diagnosis.index, 
                        values=gender_diagnosis['No'],
                        name="Not Diagnosed",
                        marker_colors=['#2ecc71', '#3498db']),
                1, 2)
    # Update layout
    fig.update_layout(
        title_text="Gender Distribution by Diagnosis Status",
        title_x=0.5,
        width=900,
        height=400,
        showlegend=True
    )
    graph2_html = pio.to_html(fig, full_html=False)
    return graph2_html

def education_distribution():
    # Calculate diagnosis rate by education level
    edu_analysis = df.groupby('Education Level')["Alzheimer’s Diagnosis"].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
    edu_analysis.columns = ['Education Level', 'Diagnosis Rate (%)']
    # Sort by diagnosis rate
    edu_analysis = edu_analysis.sort_values('Diagnosis Rate (%)', ascending=True)
    # Create Plotly horizontal bar chart
    fig = px.bar(edu_analysis, 
                y='Education Level', 
                x='Diagnosis Rate (%)',
                orientation='h',
                title='Diagnosis Rate by Education Level',
                color='Diagnosis Rate (%)',
                color_continuous_scale='Viridis')
    fig.update_layout(
        title_x=0.5,
        plot_bgcolor='white',
        showlegend=False,
        width=900,
        height=600,
        yaxis_title="Years of Education"
    )
    fig.update_traces(
        texttemplate='%{x:.1f}%',
        textposition='outside'
    )
    fig.update_xaxes(title_font=dict(size=12), tickfont=dict(size=10))
    fig.update_yaxes(title_font=dict(size=12), tickfont=dict(size=10))
    graph3_html = pio.to_html(fig, full_html=False)
    return graph3_html

def geographical_distribution():
    # Calculate diagnosis rate by country
    country_stats = df.groupby('Country')['Alzheimer’s Diagnosis'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
    country_stats.columns = ['Country', 'Diagnosis Rate (%)']
    # Sort by diagnosis rate
    country_stats = country_stats.sort_values('Diagnosis Rate (%)', ascending=False)
    # Create Plotly choropleth map
    fig = px.choropleth(country_stats,
                        locations='Country',
                        locationmode='country names',
                        color='Diagnosis Rate (%)',
                        title='Alzheimer\'s Diagnosis Rate by Country',
                        color_continuous_scale='Viridis')
    fig.update_layout(
        title_x=0.5,
        width=900,
        height=500,
        geo=dict(showframe=False,
                showcoastlines=True,
                projection_type='equirectangular'),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    graph4_html = pio.to_html(fig, full_html=False)
    return graph4_html




#Analysis Pages connectivity
@app.route('/demographic_analysis')
def demographics():
    graph1 = age_distribution()
    graph2 = gender_distribution()
    graph3 = education_distribution()
    graph4 = geographical_distribution()
    return render_template('demographic_analysis.html', graph1=graph1, graph2=graph2, graph3=graph3, graph4=graph4)




if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

