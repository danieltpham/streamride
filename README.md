# 🚕 NYC Taxi Stream — Real-Time Anomaly Detection & Drift Monitoring

A lightweight, production-style simulation of real-time machine learning on NYC taxi trip data. This system performs live anomaly detection and drift-aware model scoring via adaptive online learners — visualised on an interactive dashboard.

---

## 🎯 Project Objectives

- Simulate a real-time taxi trip stream
- Score per-trip anomalies using live model predictions
- Track and visualise concept drift (distributional shift)
- Provide observability and test coverage suitable for production ML

---

## 📦 Tech Stack

| Area             | Tools Used                                               |
|------------------|----------------------------------------------------------|
| **Language**     | Python 3.10+                                             |
| **Modelling**    | `river`, `scikit-learn`, custom anomaly logic            |
| **Streaming**    | `asyncio`, generators                                    |
| **Visualisation**| `Streamlit`, `Plotly`, `Mapbox`                          |
| **Drift Metrics**| KL divergence, PSI, rolling mean shift                   |
| **CI / Testing** | `pytest`, `mypy`, GitHub Actions                         |
| **Deployment**   | Docker, Streamlit Cloud / GCP Cloud Run (optional)       |

---

## 🗺 Dashboard Features

- 📍 **Live trip map**: Fading route history with hover tooltips
- 📉 **Model performance**: Real-time MAPE / error tracking
- ⚠️ **Anomaly alerts**: Per-trip scores and thresholds
- 📊 **Drift monitor**: Feature shift metrics + rolling plots

---

## 🧠 How It Works

1. A small slice of NYC taxi data is streamed row-by-row.
2. Each trip is scored by a lightweight online regression model.
3. Anomaly detection logic flags irregularities in speed, detour, duration.
4. A concept drift tracker monitors feature and target distribution shifts.
5. Streamlit visualises trip-level and aggregate behaviours in real time.

---

## 📁 Directory Structure

```bash
app/
├── stream_simulator.py    # Async stream logic
├── model.py               # River pipeline + wrappers
├── anomaly_detector.py    # Trip anomaly logic
├── drift_monitor.py       # Distributional shift trackers
├── visualisation.py       # Plotly + Streamlit plots
├── config.py              # Thresholds & tuning params
````

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/yourusername/nyc-taxi-anomaly-stream.git
cd nyc-taxi-anomaly-stream

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run dashboard.py
```

---

## 🧪 Testing

```bash
pytest tests/
mypy app/
```

---

## 📌 Notes

* The stream is simulated using static data for easy reproducibility.
* All models are trained incrementally using River for true online learning.
* The codebase is modular and type-annotated for clarity and extensibility.

---

## 🧩 Possible Extensions

* Incorporate real-time weather or traffic overlays
* Use multiple models with drift-specific retraining logic
* Add model performance alerting + Slack/webhook integration

---

## 🧑‍💻 Author

**Daniel Pham**
Full-stack Data Scientist | ML Systems | Optimisation & Visualisation
[LinkedIn](https://www.linkedin.com/in/your-link) | [Portfolio](https://yourwebsite.com)

---

## 📄 License

MIT License
