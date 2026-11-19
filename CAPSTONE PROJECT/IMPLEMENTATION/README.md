# Bayesian Network-Based Smart Home Energy Consumption Prediction and Decision Support System

An intelligent, interpretable, and probabilistic web application titled **Bayesian Network-Based Smart Home Energy Consumption Prediction and Decision Support System**. The system models household environmental and activity variables using a **Discrete Bayesian Network** (DAG) with **Tabular Conditional Probability Distributions (TabularCPD)** and performs exact probabilistic inference via **Variable Elimination**.

---

## 1. Project Overview & Features

- **Module 1: Household Data Input and Preprocessing**
  - Interactive inputs for ambient temperature (°C), occupant count, air conditioning (AC) state, major appliance loads, time of day, and weekday/weekend.
  - Automatic validation and discretization into discrete Bayesian states:
    - Temperature: Cool (<22°C), Moderate (22–30°C), Hot (>30°C).
    - Occupancy: Low (1–2), Medium (3–4), High (5+).
    - AC Usage: Off (0h), Moderate (1–4h / Eco), Heavy (5h+ / High).
    - Appliance Load: Low, Medium, High (dynamically computed from selected high/medium wattage devices).
    - Time of Day: Morning, Afternoon, Evening (Peak Demand), Night (Off-Peak).
    - Day Type: Weekday vs Weekend.
  - Quick demonstration preset buttons for instant scenario evaluation (*Summer Peak*, *Cool Night*, *Weekday Afternoon*).

- **Module 2: Bayesian Network-Based Energy Prediction**
  - Directed Acyclic Graph (DAG) built using `pgmpy` (`DiscreteBayesianNetwork`).
  - Verified mathematically with `model.check_model()`.
  - Exact probabilistic marginal inference via `VariableElimination`.
  - Calculates and displays exact percentages for **Low**, **Medium**, and **High** consumption.
  - Guarantees valid probability normalization:
    $$\sum_{k \in \{\text{Low}, \text{Medium}, \text{High}\}} P(\text{Energy} = k \mid \text{Evidence}) = 100.0\%$$
  - Selects the Maximum A Posteriori (MAP) dominant category.
  - Generates plain-English causal factor explanations.
  - Explicit disclaimer: Predicts relative demand categories, not measured physical kilowatt-hours (kWh).

- **Module 3: Decision Support & Energy-Saving Recommendations**
  - Heuristic-based recommendation engine prioritizing actionable actions.
  - Dynamic categorization: Cooling & HVAC, Grid & Peak Load Shifting, Appliance Staggering, and Standby Management.
  - Priority impact badges: *High Impact*, *Medium Impact*, *Actionable Tip*.
  - Advisory disclaimer ensuring scientific integrity without unsubstantiated savings claims.

- **Module 4: Graphical Analysis & Prediction History**
  - Local SQLite database (`energy_predictions.db`) initialized automatically on startup.
  - Full audit history of inputs, predicted category, and exact posterior probabilities.
  - Headless Matplotlib visualization generation (Category Donut Chart, Daily Stacked Volume, Time-of-Day Grouped Comparison).
  - Daily and Monthly aggregated summary tables.
  - One-click CSV report export (`/export/csv`).
  - History management (Delete single record, Clear history, Load benchmark scenarios).

---

## 2. Technology Stack (100% Free & Local)

- **Backend Logic:** Python 3.10+
- **Web Framework:** Flask
- **Bayesian Modeling & Inference:** `pgmpy` (`DiscreteBayesianNetwork`, `TabularCPD`, `VariableElimination`)
- **Data Manipulation:** `pandas`, `numpy`
- **Database:** SQLite3 (standard library, zero installation)
- **Visual Analytics:** `matplotlib` (Headless `Agg` backend)
- **Frontend:** HTML5, CSS3, Vanilla JavaScript, FontAwesome 6 icons

---

## 3. Project Directory Structure

```text
SmartHomeEnergy/
│
├── app.py                  # Flask web server, HTTP routes, API, and controllers
├── bayesian_model.py       # DAG definition, TabularCPDs, validation & VariableElimination
├── database.py             # SQLite access layer, schema init, history logs & aggregations
├── recommendations.py      # Module 3 decision support heuristics and conservation rules
├── charts.py               # Module 4 Matplotlib chart generator (Donut, Daily, Time-of-Day)
├── requirements.txt        # Python dependency manifest
├── README.md               # Complete setup, execution, and evaluation guide
│
├── templates/              # Jinja2 HTML Templates
│   ├── base.html           # Master layout with responsive navbar, flash alerts & footer
│   ├── index.html          # Module 1 Input Form, Module 2 Prediction & Module 3 Recs
│   ├── history.html        # Module 4 Prediction history table & CSV export
│   ├── analytics.html      # Module 4 Graphical charts, KPIs & daily/monthly summaries
│   └── model_info.html     # Technical reference of Bayesian DAG & architecture Q&A
│
└── static/                 # Static assets
    ├── style.css           # Modern dashboard responsive styling
    ├── script.js           # Live two-way sync, appliance load score & preset loaders
    └── charts/             # Auto-generated Matplotlib chart PNG files
```

---

## 4. Installation and Execution (Windows & VS Code)

Follow these exact steps in your Windows terminal or VS Code integrated terminal:

### Step 1: Open Terminal in the Project Directory
```powershell
cd "C:\Users\Surendhar Reddy\.gemini\antigravity\scratch\SmartHomeEnergy"
```

### Step 2: Create a Python Virtual Environment
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
*(If Windows PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first)*

### Step 4: Upgrade pip and Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Run the Application
```powershell
python app.py
```

### Step 6: Access the Dashboard
Open any web browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 5. How the Bayesian Network Works (Technical Explanation)

In artificial intelligence, a **Bayesian Network** is a directed graph where:
1. **Nodes** represent random variables (e.g., *Temperature*, *AC Usage*, *Energy Consumption*).
2. **Directed Edges (Arrows)** represent probabilistic cause-and-effect relationships.

### Causal Flow in this Project:
- **Temperature & Time of Day** $\rightarrow$ influence how likely the household is to turn on the **AC** at heavy levels.
- **Day Type (Weekday/Weekend)** $\rightarrow$ influences how many **Occupants** are at home.
- **Occupants, Day Type & Time of Day** $\rightarrow$ influence **Appliance Usage** (e.g., cooking dinner and running washing machines on weekend evenings).
- Finally, **AC Usage**, **Appliance Usage**, and **Occupancy** jointly determine the posterior probability of **Energy Consumption** being **Low**, **Medium**, or **High**.

### Mathematical Inference:
When a user enters conditions (called *Evidence* $E$), the model uses **Variable Elimination**:
$$P(\text{Energy} \mid E) = \frac{P(\text{Energy}, E)}{P(E)} = \alpha \sum_{\text{latent}} \prod \text{CPDs}$$
This sums out any unobserved variables and normalizes the results into exact percentages summing to 100%.

---

## 6. Sample Test Inputs & Expected Outputs

| Test Case Name | Input Conditions | Expected Prediction | Expected Posterior Probabilities | Key Influencing Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **High Consumption Scenario** | Temp: 36.5°C (Hot)<br>Occupants: 5 (High)<br>AC: Heavy<br>Appliances: High Load<br>Time: Evening<br>Day: Weekend | **HIGH** | Low: ~2.0%<br>Med: ~18.0%<br>High: **~80.0%** | Heavy AC runtime during peak evening grid hours with full household occupancy. |
| **Low Consumption Scenario** | Temp: 18.0°C (Cool)<br>Occupants: 1 (Low)<br>AC: Off<br>Appliances: Low Load<br>Time: Night<br>Day: Weekday | **LOW** | Low: **~85.0%**<br>Med: ~12.0%<br>High: ~3.0% | Inactive AC, single occupant sleeping at night with zero heavy appliances running. |
| **Medium Consumption Scenario** | Temp: 27.0°C (Moderate)<br>Occupants: 3 (Med)<br>AC: Moderate<br>Appliances: Medium Load<br>Time: Afternoon<br>Day: Weekday | **MEDIUM** | Low: ~20.0%<br>Med: **~62.0%**<br>High: ~18.0% | Balanced routine daytime power demand without extreme heating or peak concurrency. |

---

## 7. How to Demonstrate the Four Modules

### Module 1: Household Data Input and Preprocessing
1. Open the dashboard at `http://127.0.0.1:5000`.
2. Move the **Ambient Temperature** slider: observe the badge updating in real-time from *Cool* to *Moderate* to *Hot*.
3. Click checkboxes under **Active Household Appliances**: observe how selecting the *Geyser (2kW)* and *Washing Machine (1.5kW)* automatically recalculates the appliance load badge to *High Appliance Load*.
4. Click the **"Summer Peak Evening"** preset button to test instant pre-filling.

### Module 2: Bayesian Network-Based Energy Prediction
1. Click **"Compute Bayesian Energy Prediction"**.
2. Point out the **Model Valid (pgmpy)** green badge, proving that `model.check_model()` verified all CPDs.
3. Highlight the **Posterior Marginal Probabilities**: show the exact percentages (e.g., Low: 2.0%, Medium: 18.0%, High: 80.0%) and the **Sum of Probabilities: 100.0%** verification.
4. Read the **Bayesian Causal Factor Analysis** section, explaining how the network traced the impact of the Heavy AC and Peak Evening hours.
5. Emphasize the notice indicating the system predicts categorical demand tiers rather than arbitrary kWh numbers.

### Module 3: Decision Support and Energy-Saving Recommendations
1. Scroll to the **Decision Support & Conservation Tips** card.
2. Show the prioritized recommendations with badges (*High Impact*, *Medium Impact*, *Actionable Tip*).
3. Demonstrate relevance: since AC was Heavy, the top tip advises setting the thermostat to 24°C–26°C and engaging Eco mode.
4. Show the peak load-shifting recommendation advising moving laundry cycles away from 17:00–22:00.

### Module 4: Graphical Analysis and Prediction History
1. Click **"Prediction History"** in the top navigation bar:
   - Review the saved row in the SQLite database showing exact timestamp, inputs, predicted category, and mini probability bars.
   - Click the **"Download CSV Report"** button: verify downloading `.csv` file.
2. Click **"Graphical Analytics"** in the top navigation bar:
   - Review the 3 server-rendered **Matplotlib charts**: Category Distribution Donut Chart, Daily Trend Stacked Chart, and Time-of-Day Breakdown.
   - Review the **Daily Prediction Summaries** and **Monthly Prediction Summaries** tables.
3. Click **"Bayesian Model"** in the menu to review the full theoretical DAG matrix and evaluation Q&A.

---

## 8. Troubleshooting

- **Port 5000 is already in use:**
  Run the application on a different port by setting the `PORT` environment variable:
  ```powershell
  $env:PORT="5050"; python app.py
  ```
- **Matplotlib Font Warning on Windows:**
  A minor warning like `findfont: Failed to find font weight...` is normal when Matplotlib loads Segoe UI on Windows for the first time; it does not affect chart generation.
- **Database locked:**
  If SQLite reports a busy or locked state, ensure no external database viewers have opened the file with exclusive locks.
