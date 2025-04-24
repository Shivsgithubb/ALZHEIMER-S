from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib.pyplot as plt

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


#graphs connectivity
@app.route('/demographic_analysis')
def demographics():


    #1st graph

    # Defining age bins and labels
    age_bins = [49, 59, 69, 79, 89, 99, 109]
    age_labels = ['50-59', '60-69', '70-79', '80-89', '90-99', '100+']
    df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

    # Calculating Alzheimer's diagnosis rate per age group
    diagnosis_rate = (
        df[df["Alzheimer’s Diagnosis"] == "Yes"]
        .groupby('Age Group')
        .size() / df.groupby('Age Group').size()
    ) * 100

    # Plotting the diagnosis rate
    plt.figure(figsize=(10, 6))
    diagnosis_rate.plot(kind='bar', color='blue', edgecolor='black')
    plt.title("Alzheimer's Diagnosis Rate by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Diagnosis Rate (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('static/charts/rate_by_age_group.png')

    #2nd graph

    # Create a crosstab to count gender distribution by diagnosis status
    gender_distribution = pd.crosstab(df['Gender'], df["Alzheimer’s Diagnosis"])

    # Plot the pie charts
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Pie chart for diagnosed ("Yes")
    axes[0].pie(
        gender_distribution["Yes"],
        labels=gender_distribution.index,
        autopct='%1.1f%%',
        colors=['lightcoral', 'lightskyblue'],
        startangle=90
    )
    axes[0].set_title("Gender Distribution (Diagnosed)")

    # Pie chart for non-diagnosed ("No")
    axes[1].pie(
        gender_distribution["No"],
        labels=gender_distribution.index,
        autopct='%1.1f%%',
        colors=['lightcoral', 'lightskyblue'],
        startangle=90
    )
    axes[1].set_title("Gender Distribution (Non-Diagnosed)")

    plt.tight_layout()
    plt.savefig('static/charts/Gender_distribution.png')

    #3rd graph

    # Calculate diagnosis rate by education level
    edu_analysis = df.groupby('Education Level')["Alzheimer’s Diagnosis"].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
    edu_analysis.columns = ['Education Level', 'Diagnosis Rate (%)']

    # Sort by diagnosis rate
    edu_analysis = edu_analysis.sort_values('Diagnosis Rate (%)', ascending=True)

    # Simple horizontal bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(edu_analysis['Education Level'], 
                edu_analysis['Diagnosis Rate (%)'], 
                color='lightblue',
                height=0.6)

    # Add value labels
    for i, v in enumerate(edu_analysis['Diagnosis Rate (%)']):    
        plt.text(v + 0.5, i, f'{v:.1f}%', va='center')

    plt.title('Diagnosis Rate by Education Level')
    plt.xlabel('Percentage (%)')

    # Remove top and right borders
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Add light grid
    plt.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print the numbers
    print('\nDiagnosis rates by education level:')
    print(edu_analysis)
    plt.savefig('static/charts/Diagnosis_rate_by_education.png')

    #4th graph


    return render_template('demographic_analysis.html', graphs=['rate_by_age_group.png,gender_distribution.png,Diagnosis_rate_by_education.png'])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

