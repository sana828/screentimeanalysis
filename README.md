
# 📱 Screen Time Analysis Using Python

Analyze, visualize, and optimize your digital behavior with this comprehensive screen time analysis tool powered by Python. Gain insights into app usage patterns, behavioral trends, and even receive smart wellness recommendations!

---

## 📌 Features

- 📊 **Interactive Visualizations** using `matplotlib` and `seaborn`
- 🧠 **App Clustering** based on usage metrics
- 🔍 **Weekly and Daily Trends** in screen time
- 📈 **Future Usage Prediction** with Linear Regression
- 🔔 **Notification Influence Analysis**
- 🧘 **Digital Wellness Scoring System**
- 🧠 **Behavioral Change Detection**
- 🧠 **Smart Recommendations Engine**
- 🕹️ **Gamification Layer** for user engagement
- 🧾 **PDF Report Generation** summarizing key insights
- 📂 **App Category Tagging** with category-wise usage breakdown

---

## 📂 Dataset

The analysis is based on `screentime_analysis.csv`, which includes:

| Column            | Description                              |
|-------------------|------------------------------------------|
| Date              | Timestamp of app usage                   |
| App               | Name of the application                  |
| Usage (minutes)   | Duration spent on the app (in minutes)   |
| Notifications     | Notifications received from the app     |
| Times Opened      | Number of times app was opened           |

---

## 🧪 Installation

```bash
pip install pandas matplotlib seaborn scikit-learn fpdf pyttsx3
```

---

## 🚀 How to Run

```bash
python screen_time_analysis_using_pyhton.py
```

Ensure `screentime_analysis.csv` is in the same directory.

---

## 📉 Sample Outputs

- **Top 5 App Usage Trends Over Time**
- **App Clustering (via KMeans + PCA)**
- **Average Usage by Weekday**
- **Notification-to-Usage Ratios**
- **Digital Wellness Scores**
- **Smart Recommendations (e.g., mute notifications, reduce app opens)**

---

## 🧠 Wellness & Behavioral Intelligence

The project provides:
- 📈 Trend Analysis (weekly/daily spikes or dips)
- 🧮 Wellness Score = 100 - (Weighted Usage + Notifications + Opens - Efficiency)
- 🗣️ Audio Recommendations (using `pyttsx3`)
- 🧾 PDF Summary with daily averages, top apps, and scores

---

## 🧠 Smart Recommendations – Examples

> 🔔 *Consider muting notifications for Instagram. Too many alerts, not enough screen time.*

> 📵 *You open Snapchat a lot but don’t use it much. Try limiting opens or setting a timer.*

---

## 🏆 Gamification Layer

Each day is scored and awarded a **badge**:

| Badge           | Criteria Met                        |
|----------------|--------------------------------------|
| 🚨 Distracted   | Low points due to overuse or alerts |
| ⚖️ Balanced     | Moderate usage & good efficiency     |
| 🏅 Efficient     | High efficiency & healthy usage     |

---

## 📊 PDF Report Example

- Most used app
- Average usage and notifications
- Efficiency scores
- All metrics summarized in a downloadable `screen_time_report.pdf`

---

## 📂 Folder Structure

```bash
.
├── screen_time_analysis_using_pyhton.py
├── screentime_analysis.csv
├── screen_time_report.pdf
└── README.md
```

---

## 📌 Future Improvements

- Integrate Streamlit for a web-based dashboard
- Export insights to Excel or Google Sheets
- Real-time screen time tracker with mobile API (TBD)

---

## 👨‍💻 Author

**Priyanshu Sethi**  
[GitHub](https://github.com/PRIYANSHUSETHI)
