"""
Bayesian Network Model for Smart Home Energy Consumption Prediction
===================================================================
Module 2: Implements a Directed Acyclic Graph (DAG) and Tabular Conditional
Probability Distributions (TabularCPD) using pgmpy.

Variables in the network:
  1. Temperature       (States: Cool, Moderate, Hot)
  2. Day_Type          (States: Weekday, Weekend)
  3. Time_Of_Day       (States: Morning, Afternoon, Evening, Night)
  4. Occupancy         (States: Low, Medium, High)
  5. AC_Usage          (States: Off, Moderate, Heavy)
  6. Appliance_Usage   (States: Low, Medium, High)
  7. Energy_Consumption (States: Low, Medium, High) - Target Variable

Dependencies (Edges):
  - Temperature      -> AC_Usage
  - Time_Of_Day      -> AC_Usage
  - Day_Type         -> Occupancy
  - Occupancy        -> Appliance_Usage
  - Day_Type         -> Appliance_Usage
  - Time_Of_Day      -> Appliance_Usage
  - AC_Usage         -> Energy_Consumption
  - Appliance_Usage  -> Energy_Consumption
  - Occupancy        -> Energy_Consumption
"""

import numpy as np
import warnings

# Suppress minor future warnings from dependencies
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianModel
except ImportError:
    from pgmpy.models import BayesianNetwork as BayesianModel

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def build_bayesian_network():
    """
    Constructs, parameterizes, and validates the Smart Home Bayesian Network.
    Returns:
        model: Fully parameterized and validated BayesianModel instance.
        infer: VariableElimination inference engine.
    """
    model = BayesianModel([
        ('Temperature', 'AC_Usage'),
        ('Time_Of_Day', 'AC_Usage'),
        ('Day_Type', 'Occupancy'),
        ('Occupancy', 'Appliance_Usage'),
        ('Day_Type', 'Appliance_Usage'),
        ('Time_Of_Day', 'Appliance_Usage'),
        ('AC_Usage', 'Energy_Consumption'),
        ('Appliance_Usage', 'Energy_Consumption'),
        ('Occupancy', 'Energy_Consumption')
    ])

    # 1. Temperature CPD (Prior)
    # States: Cool (<22°C), Moderate (22-30°C), Hot (>30°C)
    cpd_temperature = TabularCPD(
        variable='Temperature',
        variable_card=3,
        values=[[0.25], [0.50], [0.25]],
        state_names={'Temperature': ['Cool', 'Moderate', 'Hot']}
    )

    # 2. Day_Type CPD (Prior)
    # States: Weekday, Weekend
    cpd_day_type = TabularCPD(
        variable='Day_Type',
        variable_card=2,
        values=[[0.70], [0.30]],
        state_names={'Day_Type': ['Weekday', 'Weekend']}
    )

    # 3. Time_Of_Day CPD (Prior)
    # States: Morning (06-12), Afternoon (12-17), Evening (17-22), Night (22-06)
    cpd_time_of_day = TabularCPD(
        variable='Time_Of_Day',
        variable_card=4,
        values=[[0.25], [0.25], [0.25], [0.25]],
        state_names={'Time_Of_Day': ['Morning', 'Afternoon', 'Evening', 'Night']}
    )

    # 4. Occupancy CPD (Conditional on Day_Type)
    # Parents: Day_Type (Weekday, Weekend)
    # On weekdays, homes are more likely to have lower occupancy during work/school hours.
    # On weekends, occupancy is shifted toward Medium and High.
    cpd_occupancy = TabularCPD(
        variable='Occupancy',
        variable_card=3,
        values=[
            # Weekday, Weekend
            [0.55, 0.20],  # Low
            [0.35, 0.45],  # Medium
            [0.10, 0.35]   # High
        ],
        evidence=['Day_Type'],
        evidence_card=[2],
        state_names={
            'Occupancy': ['Low', 'Medium', 'High'],
            'Day_Type': ['Weekday', 'Weekend']
        }
    )

    # 5. AC_Usage CPD (Conditional on Temperature and Time_Of_Day)
    # Parents: Temperature (Cool, Moderate, Hot), Time_Of_Day (Morning, Afternoon, Evening, Night)
    # Total parent combinations: 3 x 4 = 12 columns
    ac_columns = [
        # Cool: Morning, Afternoon, Evening, Night
        [0.92, 0.08, 0.00],  # Cool, Morning
        [0.80, 0.18, 0.02],  # Cool, Afternoon
        [0.85, 0.15, 0.00],  # Cool, Evening
        [0.95, 0.05, 0.00],  # Cool, Night
        # Moderate: Morning, Afternoon, Evening, Night
        [0.65, 0.30, 0.05],  # Moderate, Morning
        [0.30, 0.55, 0.15],  # Moderate, Afternoon
        [0.35, 0.50, 0.15],  # Moderate, Evening
        [0.55, 0.40, 0.05],  # Moderate, Night
        # Hot: Morning, Afternoon, Evening, Night
        [0.20, 0.55, 0.25],  # Hot, Morning
        [0.05, 0.35, 0.60],  # Hot, Afternoon
        [0.08, 0.42, 0.50],  # Hot, Evening
        [0.15, 0.55, 0.30]   # Hot, Night
    ]
    ac_values = np.array(ac_columns).T.tolist()
    cpd_ac_usage = TabularCPD(
        variable='AC_Usage',
        variable_card=3,
        values=ac_values,
        evidence=['Temperature', 'Time_Of_Day'],
        evidence_card=[3, 4],
        state_names={
            'AC_Usage': ['Off', 'Moderate', 'Heavy'],
            'Temperature': ['Cool', 'Moderate', 'Hot'],
            'Time_Of_Day': ['Morning', 'Afternoon', 'Evening', 'Night']
        }
    )

    # 6. Appliance_Usage CPD (Conditional on Occupancy, Day_Type, Time_Of_Day)
    # Parents: Occupancy (3), Day_Type (2), Time_Of_Day (4)
    # Total parent combinations: 3 x 2 x 4 = 24 columns
    occ_states = ['Low', 'Medium', 'High']
    day_states = ['Weekday', 'Weekend']
    tod_states = ['Morning', 'Afternoon', 'Evening', 'Night']

    app_cols = []
    for occ_i, _ in enumerate(occ_states):
        for day_i, _ in enumerate(day_states):
            for _, tod in enumerate(tod_states):
                # Peak appliance activity occurs during morning breakfast and evening dinner/leisure.
                # Night is lowest. Weekend has extra laundry/cooking load.
                tod_weight = {'Morning': 0.55, 'Afternoon': 0.40, 'Evening': 0.85, 'Night': 0.10}[tod]
                occ_weight = occ_i * 0.40
                day_weight = day_i * 0.20
                load = occ_weight + day_weight + tod_weight

                if load < 0.50:
                    probs = [0.80, 0.17, 0.03]
                elif load < 0.90:
                    probs = [0.45, 0.45, 0.10]
                elif load < 1.30:
                    probs = [0.20, 0.55, 0.25]
                else:
                    probs = [0.05, 0.35, 0.60]
                app_cols.append(probs)

    app_values = np.array(app_cols).T.tolist()
    cpd_appliance = TabularCPD(
        variable='Appliance_Usage',
        variable_card=3,
        values=app_values,
        evidence=['Occupancy', 'Day_Type', 'Time_Of_Day'],
        evidence_card=[3, 2, 4],
        state_names={
            'Appliance_Usage': ['Low', 'Medium', 'High'],
            'Occupancy': occ_states,
            'Day_Type': day_states,
            'Time_Of_Day': tod_states
        }
    )

    # 7. Energy_Consumption CPD (Target Variable)
    # Parents: AC_Usage (3), Appliance_Usage (3), Occupancy (3)
    # Total parent combinations: 3 x 3 x 3 = 27 columns
    # AC is the heaviest single load (~50% influence), followed by other appliances (~35%) and baseline occupancy (~15%)
    ac_states = ['Off', 'Moderate', 'Heavy']
    app_states = ['Low', 'Medium', 'High']
    energy_cols = []

    for ac_i, _ in enumerate(ac_states):
        for app_i, _ in enumerate(app_states):
            for occ_i, _ in enumerate(occ_states):
                score = (0.52 * (ac_i / 2.0)) + (0.33 * (app_i / 2.0)) + (0.15 * (occ_i / 2.0))
                if score <= 0.15:
                    p = [0.85, 0.12, 0.03]
                elif score <= 0.35:
                    p = [0.60, 0.32, 0.08]
                elif score <= 0.55:
                    p = [0.25, 0.55, 0.20]
                elif score <= 0.75:
                    p = [0.10, 0.40, 0.50]
                else:
                    p = [0.02, 0.18, 0.80]
                energy_cols.append(p)

    energy_values = np.array(energy_cols).T.tolist()
    cpd_energy = TabularCPD(
        variable='Energy_Consumption',
        variable_card=3,
        values=energy_values,
        evidence=['AC_Usage', 'Appliance_Usage', 'Occupancy'],
        evidence_card=[3, 3, 3],
        state_names={
            'Energy_Consumption': ['Low', 'Medium', 'High'],
            'AC_Usage': ac_states,
            'Appliance_Usage': app_states,
            'Occupancy': occ_states
        }
    )

    # Add all CPDs to the Bayesian Network
    model.add_cpds(
        cpd_temperature,
        cpd_day_type,
        cpd_time_of_day,
        cpd_occupancy,
        cpd_ac_usage,
        cpd_appliance,
        cpd_energy
    )

    # Validate conditional probability distributions consistency
    if not model.check_model():
        raise ValueError("Bayesian Network model validation failed: CPDs are inconsistent with DAG topology.")

    infer = VariableElimination(model)
    return model, infer


# Initialize global model and inference engine
BAYESIAN_MODEL, INFERENCE_ENGINE = build_bayesian_network()


def preprocess_household_data(temperature_raw, occupants_raw, ac_usage_raw, appliance_usage_raw, time_of_day_raw, day_type_raw, appliances_list=None):
    """
    Validates and preprocesses raw user inputs into Bayesian Network discrete states.
    Returns:
        clean_inputs: dict of numerical / human-readable inputs
        evidence: dict of discrete states ready for Bayesian inference
    """
    # 1. Temperature Preprocessing
    try:
        temp_val = float(temperature_raw)
    except (ValueError, TypeError):
        temp_val = 26.0

    # Clamp to realistic environmental range (-10°C to 55°C)
    temp_val = max(-10.0, min(55.0, temp_val))

    if temp_val < 22.0:
        temp_cat = 'Cool'
    elif temp_val <= 30.0:
        temp_cat = 'Moderate'
    else:
        temp_cat = 'Hot'

    # 2. Occupants Preprocessing
    try:
        occ_val = int(occupants_raw)
    except (ValueError, TypeError):
        occ_val = 3

    occ_val = max(1, min(20, occ_val))

    if occ_val <= 2:
        occ_cat = 'Low'
    elif occ_val <= 4:
        occ_cat = 'Medium'
    else:
        occ_cat = 'High'

    # 3. AC Usage Preprocessing
    valid_ac = ['Off', 'Moderate', 'Heavy']
    ac_cat = ac_usage_raw if ac_usage_raw in valid_ac else 'Moderate'

    # 4. Appliance Usage Preprocessing
    valid_app = ['Low', 'Medium', 'High']
    if appliance_usage_raw in valid_app:
        app_cat = appliance_usage_raw
    else:
        # Calculate from selected appliances list if provided
        app_list = appliances_list or []
        # Heavy power appliances: geyser, washing machine, oven, dishwasher
        high_power = {'Geyser / Water Heater', 'Washing Machine', 'Electric Oven / Microwave', 'Dishwasher'}
        # Medium power appliances: Refrigerator, Desktop PC / TV
        medium_power = {'Refrigerator', 'Television / Entertainment', 'Desktop PC / Monitor'}
        
        weight = 0.0
        for item in app_list:
            if item in high_power:
                weight += 2.0
            elif item in medium_power:
                weight += 1.0
            else:
                weight += 0.5

        if weight <= 1.5:
            app_cat = 'Low'
        elif weight <= 4.0:
            app_cat = 'Medium'
        else:
            app_cat = 'High'

    # 5. Time of Day Preprocessing
    valid_tod = ['Morning', 'Afternoon', 'Evening', 'Night']
    tod_cat = time_of_day_raw if time_of_day_raw in valid_tod else 'Evening'

    # 6. Day Type Preprocessing
    valid_day = ['Weekday', 'Weekend']
    day_cat = day_type_raw if day_type_raw in valid_day else 'Weekday'

    clean_inputs = {
        'temperature_val': temp_val,
        'temperature_cat': temp_cat,
        'occupants_val': occ_val,
        'occupants_cat': occ_cat,
        'ac_usage': ac_cat,
        'appliance_usage': app_cat,
        'time_of_day': tod_cat,
        'day_type': day_cat,
        'appliances_list': appliances_list or []
    }

    evidence = {
        'Temperature': temp_cat,
        'Occupancy': occ_cat,
        'AC_Usage': ac_cat,
        'Appliance_Usage': app_cat,
        'Time_Of_Day': tod_cat,
        'Day_Type': day_cat
    }

    return clean_inputs, evidence


def predict_energy_consumption(evidence):
    """
    Executes Bayesian inference using Variable Elimination.
    Computes exact posterior marginal distribution:
        P(Energy_Consumption | Evidence)

    Returns:
        result: dict containing probabilities, predicted_category, and explanation.
    """
    # Filter evidence to ensure valid nodes only
    valid_evidence = {k: v for k, v in evidence.items() if k in BAYESIAN_MODEL.nodes()}

    # Perform Variable Elimination inference
    query_result = INFERENCE_ENGINE.query(
        variables=['Energy_Consumption'],
        evidence=valid_evidence,
        show_progress=False
    )

    states = query_result.state_names['Energy_Consumption']
    prob_values = query_result.values

    # Convert to percentage format and verify sum
    raw_probs = {state: float(prob_values[i]) for i, state in enumerate(states)}
    low_pct = round(raw_probs.get('Low', 0.0) * 100.0, 2)
    med_pct = round(raw_probs.get('Medium', 0.0) * 100.0, 2)
    high_pct = round(raw_probs.get('High', 0.0) * 100.0, 2)

    # Adjust rounding if necessary so they sum to exactly 100%
    diff = round(100.0 - (low_pct + med_pct + high_pct), 2)
    if diff != 0:
        med_pct = round(med_pct + diff, 2)

    probabilities = {
        'Low': low_pct,
        'Medium': med_pct,
        'High': high_pct
    }

    # Determine highest posterior probability category (Maximum A Posteriori / MAP)
    predicted_category = max(probabilities, key=probabilities.get)

    # Generate plain-language causal reasoning explanation
    explanation_parts = []
    if evidence.get('AC_Usage') == 'Heavy':
        explanation_parts.append("Heavy Air Conditioning operation is a dominant contributor to high power demand.")
    elif evidence.get('AC_Usage') == 'Off':
        explanation_parts.append("The absence of active AC cooling significantly lowers base household electricity consumption.")

    if evidence.get('Appliance_Usage') == 'High':
        explanation_parts.append("Multiple concurrent major household appliances place a heavy inductive/resistive load on the electrical circuit.")
    elif evidence.get('Appliance_Usage') == 'Low':
        explanation_parts.append("Minimal appliance utilization keeps power consumption near standby baseline.")

    if evidence.get('Time_Of_Day') == 'Evening':
        explanation_parts.append("Evening hours typically represent peak household electricity demand due to simultaneous lighting, cooling, and dining.")
    elif evidence.get('Time_Of_Day') == 'Night':
        explanation_parts.append("Nighttime conditions naturally coincide with reduced household activity and appliance shutdown.")

    if evidence.get('Occupancy') == 'High':
        explanation_parts.append(f"High resident occupancy ({evidence.get('Occupancy')} level) multiplies lighting, device charging, and room cooling requirements.")

    if not explanation_parts:
        explanation_parts.append("Household conditions are balanced across standard operational baselines.")

    explanation_text = " ".join(explanation_parts)

    return {
        'probabilities': probabilities,
        'predicted_category': predicted_category,
        'explanation': explanation_text,
        'model_valid': True
    }
