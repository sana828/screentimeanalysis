# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Load Data
file_path = "screentime_analysis.csv"
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])

# Daily Usage Summary
daily_summary = df.groupby('Date').agg({
    'Usage (minutes)': 'sum',
    'App': pd.Series.nunique
}).rename(columns={'App': 'Unique Apps'})

# App-wise Aggregation
app_summary = df.groupby('App').agg({
    'Usage (minutes)': ['sum', 'mean'],
    'Notifications': ['sum', 'mean'],
    'Times Opened': ['sum', 'mean']
})

# Usage Trends Over Time
top_apps = df.groupby('App')['Usage (minutes)'].sum().nlargest(5).index
trend_df = df[df['App'].isin(top_apps)]

plt.figure(figsize=(12, 6))
sns.lineplot(data=trend_df, x='Date', y='Usage (minutes)', hue='App')
plt.title('Usage Trend Over Time for Top 5 Apps')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# App Usage Clustering
from sklearn.cluster import KMeans

features = df.groupby('App')[['Usage (minutes)', 'Notifications', 'Times Opened']].mean()
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(features)
features['Cluster'] = clusters

sns.pairplot(features, hue='Cluster', palette='tab10')
plt.suptitle('App Clustering Based on Usage Patterns', y=1.02)
plt.show()

app_analysis = df.groupby('App').agg(
    avg_usage=('Usage (minutes)', 'mean'),
    avg_notifications=('Notifications', 'mean'),
    avg_times_opened=('Times Opened', 'mean')
).reset_index()

app_analysis = app_analysis.sort_values(by='avg_usage', ascending=False)

app_analysis

df['Day of Week'] = df['Date'].dt.day_name()

weekly_usage = df.groupby('Day of Week')['Usage (minutes)'].mean().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

plt.figure(figsize=(12, 6))

ax = sns.barplot(x=weekly_usage.index, y=weekly_usage.values, palette="crest")

for p in ax.patches:
    ax.annotate(format(p.get_height(), '.1f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center',
                xytext = (0, 9), textcoords = 'offset points')

plt.title('Average Screen Time Usage per Day of the Week', fontsize=16)
plt.ylabel('Average Usage (minutes)', fontsize=12)
plt.xlabel('Day of the Week', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()
plt.show()

top_apps_data = df[df['App'].isin(['Instagram', 'Netflix', 'WhatsApp'])]

daily_app_usage = top_apps_data.groupby(['App', 'Day of Week'])['Usage (minutes)'].mean().reindex(
    pd.MultiIndex.from_product([['Instagram', 'Netflix', 'WhatsApp'],
                                ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']],
                               names=['App', 'Day of Week'])).reset_index()

plt.figure(figsize=(12, 6))

ax = sns.barplot(x='Day of Week', y='Usage (minutes)', hue='App', data=daily_app_usage, palette='Set2')

for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.1f}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center',
                    xytext=(0, 8), textcoords='offset points')

plt.title('Average Daily Usage for Instagram, Netflix, and WhatsApp', fontsize=16)
plt.ylabel('Average Usage (minutes)', fontsize=12)
plt.xlabel('Day of the Week', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='App', fontsize=10, title_fontsize=12)

plt.tight_layout()
plt.show()

notifications_data = df[df['Notifications'] > 0]

app_opened_when_notif = notifications_data.groupby('App').apply(
    lambda x: (x['Times Opened'] > 0).sum() / len(x)
).reset_index(name='Probability of Open with Notification')

app_opened_when_notif

# Attention Grabber Apps
df['Notif_per_Min'] = df['Notifications'] / df['Usage (minutes)']
attention_grabbers = df.groupby('App')['Notif_per_Min'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
attention_grabbers.plot(kind='bar', color='purple')
plt.title('Apps with Highest Notification per Minute Rate')
plt.ylabel('Notifications per Minute')
plt.tight_layout()
plt.show()

# Weekly Trends
df['Weekday'] = df['Date'].dt.day_name()
df['Week'] = df['Date'].dt.isocalendar().week

weekly_usage = df.groupby('Week')["Usage (minutes)"].sum()

plt.figure(figsize=(10, 5))
weekly_usage.plot(marker='o')
plt.title('Weekly Total Screen Time')
plt.xlabel('Week Number')
plt.ylabel('Total Usage (minutes)')
plt.tight_layout()
plt.show()

# Usage Spike Detection
daily_total = df.groupby('Date')['Usage (minutes)'].sum()
z_scores = (daily_total - daily_total.mean()) / daily_total.std()
spikes = z_scores[z_scores > 1.5]
drops = z_scores[z_scores < -1.5]
print("\nUsage Spikes:\n", spikes)
print("\nUsage Drops:\n", drops)

# Top 3 App Dashboard
app_totals = df.groupby('App')[['Usage (minutes)', 'Notifications', 'Times Opened']].sum()
top3_usage = app_totals.sort_values('Usage (minutes)', ascending=False).head(3)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
top3_usage['Usage (minutes)'].plot(kind='bar', ax=axes[0], title='Top 3 by Usage', color='green')
top3_usage['Notifications'].plot(kind='bar', ax=axes[1], title='Top 3 by Notifications', color='red')
top3_usage['Times Opened'].plot(kind='bar', ax=axes[2], title='Top 3 by Times Opened', color='blue')
plt.tight_layout()
plt.show()

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
df = pd.read_csv("screentime_analysis.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Aggregate features per app
app_features = df.groupby('App')[['Usage (minutes)', 'Notifications', 'Times Opened']].mean()

# Normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(app_features)

# Run KMeans clustering
k = 3  # Number of clusters (you can change this or use elbow method below)
kmeans = KMeans(n_clusters=k, random_state=42)
app_features['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualize clusters in 2D using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(scaled_features)
app_features['PCA1'] = reduced[:, 0]
app_features['PCA2'] = reduced[:, 1]

# Plot the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=app_features, x='PCA1', y='PCA2', hue='Cluster', s=100, palette='Set2')
for i in app_features.index:
    plt.text(app_features.loc[i, 'PCA1'] + 0.02, app_features.loc[i, 'PCA2'], i)
plt.title('App Clustering Based on Usage')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.tight_layout()
plt.show()

# View cluster assignments
print(app_features[['Usage (minutes)', 'Notifications', 'Times Opened', 'Cluster']])

# Predict Future Usage
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Daily prediction for total usage
daily = df.groupby('Date').agg({
    'Usage (minutes)': 'sum',
    'Notifications': 'sum',
    'Times Opened': 'sum'
}).reset_index()
# Ensure 'Date' column is of datetime type
daily['Date'] = pd.to_datetime(daily['Date'])  # Convert 'Date' column to datetime
daily['Day'] = (daily['Date'] - daily['Date'].min()).dt.days

X = daily[['Day', 'Notifications', 'Times Opened']]
y = daily['Usage (minutes)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nLinear Regression MSE:", mean_squared_error(y_test, y_pred))

!pip install fpdf

! pip install pyttsx3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from fpdf import FPDF
import pyttsx3

# Digital Wellness Scoring System
# Score = 100 - (usage_weight + notif_weight + open_weight - efficiency_score)

normalized = df.copy()
normalized['Usage_norm'] = (normalized['Usage (minutes)'] - df['Usage (minutes)'].min()) / (df['Usage (minutes)'].max() - df['Usage (minutes)'].min())
normalized['Notif_norm'] = (df['Notifications'] - df['Notifications'].min()) / (df['Notifications'].max() - df['Notifications'].min())
normalized['Opens_norm'] = (df['Times Opened'] - df['Times Opened'].min()) / (df['Times Opened'].max() - df['Times Opened'].min())
normalized['Efficiency'] = df['Usage (minutes)'] / df['Times Opened'].replace(0, 1)
normalized['Efficiency_norm'] = (normalized['Efficiency'] - normalized['Efficiency'].min()) / (normalized['Efficiency'].max() - normalized['Efficiency'].min())

normalized['Wellness Score'] = 100 - (normalized['Usage_norm']*30 + normalized['Notif_norm']*30 + normalized['Opens_norm']*30 - normalized['Efficiency_norm']*10)

# Behavioral Change Detection
df_sorted = df.sort_values('Date')
first_10 = df_sorted[df_sorted['Date'] <= df_sorted['Date'].unique()[9]]
last_10 = df_sorted[df_sorted['Date'] >= df_sorted['Date'].unique()[-10]]

first_avg = first_10.groupby('App')[['Usage (minutes)', 'Notifications', 'Times Opened']].mean()
last_avg = last_10.groupby('App')[['Usage (minutes)', 'Notifications', 'Times Opened']].mean()

change = last_avg - first_avg
print("\nBehavioral Change (Last 10 Days vs First 10 Days):\n")
print(change.dropna())

# Smart Recommendations Engine
recommendations = []
for app in df['App'].unique():
    app_data = df[df['App'] == app]
    avg_usage = app_data['Usage (minutes)'].mean()
    avg_notif = app_data['Notifications'].mean()
    avg_open = app_data['Times Opened'].mean()
    eff = avg_usage / avg_open if avg_open else 0
    if avg_notif > 30 and eff < 2:
        recommendations.append(f"🔔 Consider muting notifications for {app}. Too many alerts, not enough screen time.")
    if avg_open > 40 and avg_usage < 40:
        recommendations.append(f"📵 You open {app} a lot but don’t use it much. Try limiting opens or setting a timer.")

print("\nSmart Recommendations:\n")
for r in recommendations:
    print(r)

# Gamification Layer
daily_usage = df.groupby('Date')[['Usage (minutes)', 'Notifications', 'Times Opened']].sum().reset_index()
daily_usage['Efficiency'] = daily_usage['Usage (minutes)'] / daily_usage['Times Opened'].replace(0, 1)
daily_usage['Points'] = 0
daily_usage.loc[daily_usage['Usage (minutes)'] < 120, 'Points'] += 20
daily_usage.loc[daily_usage['Notifications'] < 25, 'Points'] += 20
daily_usage.loc[daily_usage['Efficiency'] > 2.5, 'Points'] += 20
daily_usage['Badge'] = pd.cut(daily_usage['Points'], bins=[0, 20, 40, 60], labels=["🚨 Distracted", "⚖️ Balanced", "🏅 Efficient"])

print("\nGamification Summary:\n")
print(daily_usage[['Date', 'Points', 'Badge']])

# PDF Report Generator
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
# Change the title to remove the emoji
# pdf.cell(200, 10, txt="📊 Screen Time Report", ln=1, align='C')
pdf.cell(200, 10, txt="Screen Time Report", ln=1, align='C')  # Removed the emoji
pdf.cell(200, 10, txt=f"Top Used App: {df.groupby('App')['Usage (minutes)'].sum().idxmax()}", ln=2)
pdf.cell(200, 10, txt=f"Average Daily Usage: {daily_usage['Usage (minutes)'].mean():.2f} mins", ln=3)
pdf.cell(200, 10, txt=f"Average Daily Notifications: {daily_usage['Notifications'].mean():.2f}", ln=4)
pdf.cell(200, 10, txt=f"Efficiency Score (avg): {daily_usage['Efficiency'].mean():.2f}", ln=5)
pdf.output("screen_time_report.pdf")

# App Category Tagging
category_map = {
    'Instagram': 'Social',
    'WhatsApp': 'Messaging',
    'YouTube': 'Entertainment',
    'Gmail': 'Productivity',
    'Facebook': 'Social',
    'Snapchat': 'Social'
}
df['Category'] = df['App'].map(category_map).fillna('Other')
category_usage = df.groupby('Category')['Usage (minutes)'].sum()

plt.figure(figsize=(8, 5))
category_usage.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title("App Usage by Category")
plt.ylabel("")
plt.tight_layout()
plt.show()
