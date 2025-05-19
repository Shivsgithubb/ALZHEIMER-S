from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pickle
import numpy as np
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, User
from forms import LoginForm, SignupForm

app = Flask(__name__)
app.secret_key = 'supersecretmre'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

df = pd.read_csv('alzheimers_prediction_dataset.csv')

# Load the trained model
with open('alzheimers_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    form = LoginForm()
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password')
    
    return render_template('login.html')

@app.route('/signup', methods=['POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists')
            return redirect(url_for('login'))
        
        user = User(name=name, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
        return redirect(url_for('dashboard'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

#graphs functions
def age_distribution():
    bins = [0, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ['<30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90+']
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    # Calculate diagnosis rate by age group
    age_analysis = df.groupby('Age Group')['Alzheimer’s Diagnosis'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
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
    gender_diagnosis = pd.crosstab(df['Gender'], df['Alzheimer’s Diagnosis'])
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
    edu_analysis = df.groupby('Education Level')['Alzheimer’s Diagnosis'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
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


def snoking_status():
    # First prepare the data as before
    grouped_data = df.groupby(['Smoking Status', 'Alcohol Consumption', 'Alzheimer’s Diagnosis']).size().reset_index(name='Count')

    # Create the Plotly figure
    fig = px.bar(grouped_data, 
                x='Smoking Status',
                y='Count',
                color='Alzheimer’s Diagnosis',
                barmode='group',
                color_discrete_sequence=px.colors.qualitative.Set2,
                title='Alzheimers Diagnosis by Smoking Status and Alcohol Consumption',
                labels={'Count': 'Number of Individuals',
                        'Smoking Status': 'Smoking Status',
                        'Alzheimer\'s Diagnosis': 'Diagnosis'})

    # Update layout
    fig.update_layout(
        title_x=0.5,
        plot_bgcolor='white',
        width=900,
        height=500,
        legend=dict(
            title='Alzheimer\'s Diagnosis',
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    # Update axes
    fig.update_xaxes(
        tickangle=45,
        gridcolor='lightgray'
    )
    fig.update_yaxes(
        gridcolor='lightgray'
    )

    # Add value labels on the bars
    fig.update_traces(
        texttemplate='%{y}',
        textposition='outside'
    )
    graph9_html = pio.to_html(fig, full_html=False)
    return graph9_html


def BMI_distribution():
    fig = px.box(df, x='Alzheimer’s Diagnosis', y='BMI',
             color='Alzheimer’s Diagnosis',
             color_discrete_sequence=['#1f77b4', '#ff7f0e'],
             title='BMI Distribution by Alzheimers Diagnosis',
             labels={'Alzheimers Diagnosis': 'Diagnosis'})

    # Update layout
    fig.update_layout(
        title_x=0.5,
        plot_bgcolor='white',
        showlegend=False,
        width=800,
        height=500
    )

    # Update axes
    fig.update_xaxes(gridcolor='lightgray', 
                    ticktext=['No', 'Yes'], 
                    tickvals=[0, 1])
    fig.update_yaxes(gridcolor='lightgray')

    graph10_html = pio.to_html(fig, full_html=False)
    return graph10_html

def physical_activity():
    # Create a cross-tabulation of Physical Activity Level and Alzheimer's Diagnosis
    activity_ct = pd.crosstab(df["Physical Activity Level"], df["Alzheimer’s Diagnosis"])
    
    # Calculate percentages
    activity_percentages = pd.DataFrame()
    for level in ['Low', 'Medium', 'High']:
        total = activity_ct.loc[level, 'No'] + activity_ct.loc[level, 'Yes']
        activity_percentages.loc[level, 'No (%)'] = (activity_ct.loc[level, 'No'] / total) * 100
        activity_percentages.loc[level, 'Yes (%)'] = (activity_ct.loc[level, 'Yes'] / total) * 100

    # Create figure with secondary y-axis
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('Number of Individuals by Physical Activity Level',
                                    'Percentage Distribution by Physical Activity Level'))

    # Add bars for count plot (left subplot)
    fig.add_trace(
        go.Bar(name='No', x=activity_ct.index, y=activity_ct['No'],
            text=activity_ct['No'].apply(lambda x: f'{x:,}'),
            textposition='inside',
            marker_color='#2ecc71'),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(name='Yes', x=activity_ct.index, y=activity_ct['Yes'],
            text=activity_ct['Yes'].apply(lambda x: f'{x:,}'),
            textposition='inside',
            marker_color='#e74c3c'),
        row=1, col=1
    )

    # Add bars for percentage plot (right subplot)
    fig.add_trace(
        go.Bar(name='No', x=activity_percentages.index, y=activity_percentages['No (%)'],
            text=activity_percentages['No (%)'].apply(lambda x: f'{x:.1f}%'),
            textposition='inside',
            marker_color='#2ecc71',
            showlegend=False),
        row=1, col=2
    )

    fig.add_trace(
        go.Bar(name='Yes', x=activity_percentages.index, y=activity_percentages['Yes (%)'],
            text=activity_percentages['Yes (%)'].apply(lambda x: f'{x:.1f}%'),
            textposition='inside',
            marker_color='#e74c3c',
            showlegend=False),
            row=1, col=2
        )

    # Update layout
    fig.update_layout(
        title_text="Physical Activity Levels and Alzheimer's Diagnosis Analysis",
        title_x=0.5,
        barmode='stack',
        height=600,
        width=1200,
        showlegend=True,
        legend_title="Alzheimer's Diagnosis",
        plot_bgcolor='white'
    )

    # Update axes
    fig.update_xaxes(title_text="Physical Activity Level", row=1, col=1, gridcolor='lightgray')
    fig.update_xaxes(title_text="Physical Activity Level", row=1, col=2, gridcolor='lightgray')
    fig.update_yaxes(title_text="Number of Individuals", row=1, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Percentage (%)", row=1, col=2, gridcolor='lightgray')

    graph11_html = pio.to_html(fig, full_html=False)
    return graph11_html

def Multifactor_analysis():
    # First create the age groups as before
    df['Age_Group'] = pd.cut(df['Age'], 
                            bins=[0, 50, 60, 70, 80, 90, 100],
                            labels=['<50', '50-60', '60-70', '70-80', '80-90', '90+'])

    # Create the subplot layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Age Distribution by Physical Activity Level and Diagnosis',
                    'BMI Distribution by Physical Activity Level and Diagnosis',
                    'Alzheimer\'s Diagnosis Rate by Age Group and Physical Activity Level'),
        specs=[[{}, {}],
            [{"colspan": 2}, None]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # 1. Age Distribution by Physical Activity and Diagnosis (Box Plot)
    for i, diagnosis in enumerate(['No', 'Yes']):
        for j, activity in enumerate(['Low', 'Medium', 'High']):
            subset = df[(df['Alzheimer’s Diagnosis'] == diagnosis) & 
                    (df['Physical Activity Level'] == activity)]
            
            fig.add_trace(
                go.Box(x=[activity] * len(subset),
                    y=subset['Age'],
                    name=diagnosis,
                    legendgroup=diagnosis,
                    showlegend=True if j == 0 else False,
                    marker_color='#2ecc71' if diagnosis == 'No' else '#e74c3c'),
                row=1, col=1
            )

    # 2. BMI Distribution by Physical Activity and Diagnosis (Box Plot)
    for i, diagnosis in enumerate(['No', 'Yes']):
        for j, activity in enumerate(['Low', 'Medium', 'High']):
            subset = df[(df['Alzheimer’s Diagnosis'] == diagnosis) & 
                    (df['Physical Activity Level'] == activity)]
            
            fig.add_trace(
                go.Box(x=[activity] * len(subset),
                    y=subset['BMI'],
                    name=diagnosis,
                    legendgroup=diagnosis,
                    showlegend=False,
                    marker_color='#2ecc71' if diagnosis == 'No' else '#e74c3c'),
                row=1, col=2
            )

    # 3. Diagnosis Rate by Age Group and Physical Activity (Bar Plot)
    diagnosis_by_age_activity = df.groupby(['Age_Group', 'Physical Activity Level'])['Alzheimer’s Diagnosis'].apply(
        lambda x: (x == 'Yes').mean() * 100
    ).reset_index()

    colors = px.colors.qualitative.Set2
    for i, activity in enumerate(['Low', 'Medium', 'High']):
        activity_data = diagnosis_by_age_activity[diagnosis_by_age_activity['Physical Activity Level'] == activity]
        
        fig.add_trace(
            go.Bar(x=activity_data['Age_Group'],
                y=activity_data['Alzheimer’s Diagnosis'],
                name=activity,
                text=activity_data['Alzheimer’s Diagnosis'].round(1).astype(str) + '%',
                textposition='outside',
                marker_color=colors[i]),
            row=2, col=1
        )

    # Update layout and formatting
    fig.update_layout(
        title_text='Multi-Factor Analysis of Alzheimer\'s Diagnosis',
        title_x=0.5,
        height=900,
        width=1200,
        showlegend=True,
        legend_title='Diagnosis Status',
        barmode='group',
        plot_bgcolor='white'
    )

    # Update axes labels and formatting
    fig.update_xaxes(title_text='Physical Activity Level', row=1, col=1, gridcolor='lightgray')
    fig.update_xaxes(title_text='Physical Activity Level', row=1, col=2, gridcolor='lightgray')
    fig.update_xaxes(title_text='Age Group', row=2, col=1, gridcolor='lightgray')

    fig.update_yaxes(title_text='Age', row=1, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text='BMI', row=1, col=2, gridcolor='lightgray')
    fig.update_yaxes(title_text='Diagnosis Rate (%)', row=2, col=1, gridcolor='lightgray')
    graph12_html = pio.to_html(fig, full_html=False)
    return graph12_html


def genetic_risk():
    fig = px.bar(
    df.groupby("Genetic Risk Factor (APOE-ε4 allele)")['Alzheimer’s Diagnosis'].value_counts().unstack().reset_index(),
    x="Genetic Risk Factor (APOE-ε4 allele)",
    y=["Yes", "No"],
    barmode="group",
    title="Alzheimer’s Diagnosis by Genetic Risk (APOE-ε4 allele)",
    labels={"value": "Count", "Genetic Risk Factor (APOE-ε4 allele)": "APOE-ε4 Allele"},
    template="plotly_white"
    )
    graph5_html = pio.to_html(fig, full_html=False)
    return graph5_html

def urban_vs_rural_living():
    fig = px.bar(
    df.groupby("Urban vs Rural Living")['Alzheimer’s Diagnosis'].value_counts(normalize=True).mul(100).unstack().reset_index(),
    x="Urban vs Rural Living",
    y=["Yes", "No"],
    title="Diagnosis Percentage by Urban vs Rural Living",
    labels={"value": "Percentage", "Urban vs Rural Living": "Living Area"},
    template="plotly_white"
    )   
    graph6_html = pio.to_html(fig, full_html=False)
    return graph6_html

def air_pollution_exposure_distribution():
    fig = px.box(
    df,
    x='Alzheimer’s Diagnosis',
    y='Air Pollution Exposure',
    color='Alzheimer’s Diagnosis',
    title='Air Pollution Exposure Distribution by Alzheimer\'s Diagnosis',
    template="plotly_white"
    )
    graph7_html = pio.to_html(fig, full_html=False)
    return graph7_html

def distribution_of_APOE_Dataset():
    fig = px.pie(
    df,
    names="Genetic Risk Factor (APOE-ε4 allele)",
    title="Distribution of APOE-ε4 Allele in Dataset",
    hole=0.4,
    template="plotly_white"
    )    
    graph8_html = pio.to_html(fig, full_html=False)
    return graph8_html

#SOCIOECONOMIC ANALYSIS

def Diagnosis_by_income_level():
    fig = px.bar(
    df.groupby("Income Level")['Alzheimer’s Diagnosis'].value_counts().unstack().reset_index(),
    x="Income Level",
    y=["Yes", "No"],
    barmode="group",
    title="Alzheimer’s Diagnosis by Income Level",
    labels={"value": "Count"},
    template="plotly_white"
    )
    graph13_html = pio.to_html(fig, full_html=False)
    return graph13_html

def Diagnosis_by_Employment_status():
    df_emp_grouped = df.groupby("Employment Status")['Alzheimer’s Diagnosis'].value_counts().unstack().fillna(0).reset_index()

    fig = px.bar(
    df_emp_grouped,
    x="Employment Status",
    y=["Yes", "No"],
    barmode="group",
    title="Alzheimer’s Diagnosis by Employment Status",
    labels={"value": "Count"},
    template="plotly_white"
    )
    graph14_html = pio.to_html(fig, full_html=False)
    return graph14_html

def Diagnosis_by_Social_Engagement_Level():
    df_social_grouped = df.groupby("Social Engagement Level")['Alzheimer’s Diagnosis'].value_counts().unstack().fillna(0).reset_index()
    fig = px.bar(
    df_social_grouped,
    x="Social Engagement Level",
    y=["Yes", "No"],
    barmode="group",
    title="Alzheimer’s Diagnosis by Social Engagement Level",
    labels={"value": "Count"},
    template="plotly_white"
    )
    graph15_html = pio.to_html(fig, full_html=False)
    return graph15_html

def Diagnosis_by_Marital_Status():
    fig = px.bar(
    df.groupby("Marital Status")['Alzheimer’s Diagnosis'].value_counts().unstack().reset_index(),
    x="Marital Status",
    y=["Yes", "No"],
    barmode="group",
    title="Alzheimer’s Diagnosis by Marital Status",
    template="plotly_white"
    )
    graph16_html = pio.to_html(fig, full_html=False)
    return graph16_html

#COGNITIVE PSYCHOLOGICAL FACTORS
def Cognitive_Test_Scores_analysis():
    fig = px.box(
    df,
    x='Alzheimer’s Diagnosis',
    y='Cognitive Test Score',
    color='Alzheimer’s Diagnosis',
    title='Cognitive Test Scores by Alzheimer\'s Diagnosis',
    points="all",
    template="plotly_white"
    )
    graph17_html = pio.to_html(fig, full_html=False)
    return graph17_html

def Distribution_of_Depression_Levels():
    fig = px.violin(
    df,
    y='Depression Level',
    x='Alzheimer’s Diagnosis',
    color='Alzheimer’s Diagnosis',
    box=True,
    points="all",
    title='Distribution of Depression Levels by Alzheimer\'s Diagnosis',
    template="plotly_white"
    )
    graph18_html = pio.to_html(fig, full_html=False)
    return graph18_html

def Sleep_Quality_Disrtribution():
    fig = px.histogram(
    df,
    x='Sleep Quality',
    color='Alzheimer’s Diagnosis',
    barmode="overlay",
    nbins=20,
    title='Sleep Quality Distribution by Alzheimer\'s Diagnosis',
    template="plotly_white"
    )
    graph19_html = pio.to_html(fig, full_html=False)
    return graph19_html

def Sleep_Quality_and_Alzheimers_Diagnosis():
    fig = px.sunburst(
    df,
    path=['Sleep Quality', 'Alzheimer’s Diagnosis'],
    title='Sleep Quality and Alzheimer\'s Diagnosis Breakdown',
    color='Alzheimer’s Diagnosis',
    template="plotly_white"
    )
    graph20_html = pio.to_html(fig, full_html=False)
    return graph20_html
    

#Analysis Pages connectivity
@app.route('/demographic_analysis')
def demographics():
    graph1 = age_distribution()
    graph2 = gender_distribution()
    graph3 = education_distribution()
    graph4 = geographical_distribution()
    return render_template('demographic_analysis.html', graph1=graph1, graph2=graph2, graph3=graph3, graph4=graph4)

@app.route('/Genetic_and_environmental_risk')
def genetic():
    graph5 = genetic_risk()
    graph6 = urban_vs_rural_living()
    graph7 = air_pollution_exposure_distribution()
    graph8 = distribution_of_APOE_Dataset()
    return render_template('Genetic_and_environmental_risk.html', graph5=graph5, graph6=graph6, graph7=graph7, graph8=graph8) 

@app.route('/Lifestyle_and_health_factors')
def lifestyle():
    graph9 = snoking_status()
    graph10 = BMI_distribution()
    graph11 = physical_activity()
    graph12 = Multifactor_analysis()
    return render_template('Lifestyle_and_health_factors.html', graph9=graph9, graph10=graph10, graph11=graph11, graph12=graph12)

@app.route('/socioeconomic_&_social_engagement')
def socioeconomic():
    graph13 = Diagnosis_by_income_level()
    graph14 = Diagnosis_by_Employment_status()
    graph15 = Diagnosis_by_Social_Engagement_Level()
    graph16 = Diagnosis_by_Marital_Status()
    return render_template('socioeconomic_&_social_engagement.html', graph13=graph13, graph14=graph14, graph15=graph15, graph16=graph16)

@app.route('/cognitive_phsychological_factors')
def cognitive():
    graph17 = Cognitive_Test_Scores_analysis()
    graph18 = Distribution_of_Depression_Levels()
    graph19 = Sleep_Quality_Disrtribution()
    graph20 = Sleep_Quality_and_Alzheimers_Diagnosis()

    return render_template('cognitive_phsychological_factors.html', graph17=graph17, graph18=graph18, graph19=graph19, graph20=graph20)

def predict_alzheimers_risk(sample_data):
    """
    Make prediction and return detailed results
    """
    prediction_proba = model.predict_proba(sample_data)[0, 1]
    prediction = model.predict(sample_data)[0]
    
    risk_level = ""
    recommendations = []
    
    if prediction == 1:
        if prediction_proba > 0.8:
            risk_level = "VERY HIGH RISK"
            recommendations = [
                "Immediate medical consultation is strongly advised",
                "Consider comprehensive cognitive assessment",
                "Regular monitoring of cognitive function"
            ]
        else:
            risk_level = "ELEVATED RISK"
            recommendations = [
                "Schedule a medical check-up",
                "Monitor cognitive changes",
                "Consider lifestyle modifications"
            ]
    else:
        if prediction_proba < 0.2:
            risk_level = "VERY LOW RISK"
            recommendations = [
                "Maintain current healthy lifestyle",
                "Regular exercise and mental activities",
                "Routine health check-ups"
            ]
        else:
            risk_level = "LOW RISK"
            recommendations = [
                "Continue healthy practices",
                "Monitor any cognitive changes",
                "Regular health check-ups"
            ]
    
    return {
        'prediction': 'High Risk' if prediction == 1 else 'Low Risk',
        'probability': f"{prediction_proba:.2%}",
        'risk_level': risk_level,
        'recommendations': recommendations
    }

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    # Handle POST request (form submission)
    try:
        input_data = {
            'Age': int(request.form['age']),
            'Gender': request.form['gender'],
            'Physical Activity Level': request.form['activity_level'],
            'Smoking Status': request.form['smoking_status'],
            'Family History of Alzheimer’s': request.form['family_history'],
            'Dietary Habits': request.form['dietary_habits'],
            'Air Pollution Exposure': request.form['pollution_exposure'],
            'Employment Status': request.form['employment_status'],
            'Marital Status': request.form['marital_status'],
            'Genetic Risk Factor (APOE-ε4 allele)': request.form['genetic_risk'],
            'Social Engagement Level': request.form['social_engagement'],
            'Income Level': request.form['income_level'],
            'Stress Levels': request.form['stress_level'],
            'Urban vs Rural Living': request.form['living_area']
        }
        
        # Convert to DataFrame
        sample_data = pd.DataFrame([input_data])
        
        # Make prediction
        results = predict_alzheimers_risk(sample_data)
        
        return render_template('prediction_result.html', results=results, input_data=input_data)
    
    except Exception as e:
        flash('Error processing your request. Please check your inputs and try again.', 'error')
        return redirect(url_for('predict'))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

