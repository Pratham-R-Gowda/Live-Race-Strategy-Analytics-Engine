# 🏎️ The F1 Pit Wall: Live Race Strategy & Analytics Engine

## 👋 What is this?

If you watch Formula 1, you know the race isn't just happening on the track, it's happening on the pit wall.

I wanted to see if I could build a Machine Learning pipeline that thinks like an F1 strategist. This repository is an end-to-end data science project that pulls live telemetry, predicts lap times based on track physics, automatically flags botched pit stops, and wraps it all into a live, interactive web dashboard.

**🌟 [Click here to view the live Streamlit Web App!]** _(Add your link here once deployed)_

---

## 🏗️ What's Under the Hood?

This project is broken down into three core ML engines:

### 1. The Lap Time Predictor (XGBoost)

A car's lap time is a continuous tug-of-war: **Fuel Burn** makes the car lighter and faster, while **Tire Degradation** removes grip and makes it slower.

- I engineered a feature using `LapNumber` as a proxy for fuel weight and trained an **XGBoost Regressor** to mathematically calculate this "crossover effect."
- It predicts a driver's exact lap time based on tire compound, tire age, and track temperature.

### 2. The Pit Crew Anomaly Detector (Isolation Forest)

A standard F1 pit stop takes ~24 seconds in the pit lane. But what happens if a wheel gun jams or a jack slips?

- Instead of arbitrarily guessing what a "slow" stop is, I built an **Unsupervised Anomaly Detection** model using an `Isolation Forest`.
- It analyzes hundreds of pit stops, mathematically isolates the catastrophic outliers, and builds a leaderboard grading pit crews on raw median speed and error rate.

### 3. The Live Strategy Dashboard (Streamlit)

I didn't want this to just live in a Jupyter Notebook. I exported the trained ML models using `joblib` and deployed an interactive Python web application using `Streamlit`. Users can act as the strategist, adjusting fuel loads and tire wear on the fly to see how it impacts lap times.

---

## 🧗‍♂️ The Engineering Struggles (Or, "Things that broke")

Machine learning is 80% data cleaning and 20% algorithms. Here are the biggest hurdles I hit while building this:

- **The Ergast API Shutdown:** The community standard for F1 pit stop data was recently deprecated. I had to engineer a workaround by extracting a driver's `PitInTime` on one lap and subtracting it from their `PitOutTime` on the subsequent lap using pure `FastF1` telemetry.
- **The Pandas Index Nightmare:** While merging track temperature weather data into my telemetry dataframe, the scrambled index from filtering out Safety Car laps caused a massive data-alignment crash. I had to implement strict index-reset protocols to get the matrices to compile.
- **Taming the Outliers:** When grading pit crews, using a standard `.mean()` was impossible because a single 60-second botched stop would ruin a crew's average. I implemented robust `.median()` aggregations to capture a crew's true baseline speed.

---

## 🛠️ The Tech Stack

- **Web Framework:** `Streamlit`
- **Machine Learning:** `XGBoost`, `Scikit-Learn` (Isolation Forest)
- **Data Engineering:** `FastF1` (Telemetry API), `Pandas`, `NumPy`
- **Visualization:** `Seaborn`, `Matplotlib`
- **Model Serialization:** `Joblib`

---

## 🚀 Play With It Yourself

Want to test Fernando Alonso's predicted pace on 20-lap-old Hard tires?

**1. Clone this repository:**

```bash
git clone [https://github.com/Pratham-R-Gowda/your-repo-name.git](https://github.com/Pratham-R-Gowda/your-repo-name.git)
cd your-repo-name
```
