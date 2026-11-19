"""
Decision Support and Energy-Saving Recommendations Engine
=========================================================
Module 3: Generates targeted, actionable energy conservation advice
based on predicted consumption levels (Low, Medium, High) and specific
household operating conditions.

Note:
Recommendations are advisory heuristic guidelines and do not claim measured
financial or kWh savings without dedicated metered sub-metering data.
"""

def generate_recommendations(clean_inputs, prediction_result):
    """
    Evaluates household inputs and Bayesian prediction to produce prioritized,
    practical decision-support recommendations.

    Args:
        clean_inputs (dict): Preprocessed household parameters.
        prediction_result (dict): Output from Bayesian inference.

    Returns:
        recommendations (list of dict): Prioritized recommendation objects.
        disclaimer (str): Scientific disclaimer.
    """
    recs = []
    category = prediction_result.get('predicted_category', 'Medium')
    ac_usage = clean_inputs.get('ac_usage', 'Moderate')
    temp_val = clean_inputs.get('temperature_val', 26.0)
    temp_cat = clean_inputs.get('temperature_cat', 'Moderate')
    time_of_day = clean_inputs.get('time_of_day', 'Evening')
    app_usage = clean_inputs.get('appliance_usage', 'Medium')
    appliances_list = clean_inputs.get('appliances_list', [])
    occupants_val = clean_inputs.get('occupants_val', 3)

    # 1. AC & HVAC Optimization
    if ac_usage == 'Heavy':
        recs.append({
            'title': 'Optimize AC Thermostat & Setpoint',
            'category': 'Cooling & HVAC',
            'priority': 'High Impact',
            'badge_color': 'danger',
            'icon': 'snowflake',
            'description': (
                f"With heavy AC runtime at an outdoor temperature of {temp_val}°C, setting the thermostat "
                "to 24°C–26°C instead of sub-20°C reduces compressor duty cycles significantly. "
                "Pairing AC with ceiling fans improves perceived comfort through convective airflow."
            ),
            'action': 'Adjust thermostat to 24°C+ and engage Eco/Sleep mode.'
        })
    elif ac_usage == 'Moderate' and temp_cat == 'Cool':
        recs.append({
            'title': 'Leverage Natural Ventilation',
            'category': 'Cooling & HVAC',
            'priority': 'Medium Impact',
            'badge_color': 'warning',
            'icon': 'wind',
            'description': (
                f"The ambient temperature is currently {temp_val}°C ({temp_cat}). "
                "Consider opening windows for cross-ventilation instead of active AC cooling."
            ),
            'action': 'Switch off AC and utilize ambient air circulation.'
        })
    elif ac_usage == 'Moderate' and time_of_day in ['Night']:
        recs.append({
            'title': 'Enable AC Sleep / Timer Function',
            'category': 'Cooling & HVAC',
            'priority': 'Actionable Tip',
            'badge_color': 'info',
            'icon': 'moon',
            'description': (
                "Body metabolic rates drop during sleep, requiring less active cooling. "
                "A programmed timer prevents the compressor from running continuously until morning."
            ),
            'action': 'Set a 3 to 4-hour automatic shut-off timer on cooling units.'
        })

    # 2. Peak Hours & Load Shifting
    if time_of_day == 'Evening':
        recs.append({
            'title': 'Shift Flexible High-Power Loads to Off-Peak',
            'category': 'Grid & Peak Hours',
            'priority': 'High Impact',
            'badge_color': 'danger',
            'icon': 'clock',
            'description': (
                "Evening hours (17:00–22:00) coincide with peak municipal grid demand and highest "
                "household concurrency. Postponing delayable tasks like laundry or dishwasher cycles "
                "to off-peak hours mitigates circuit stress."
            ),
            'action': 'Schedule washing machines and dishwashers for morning or late-night operation.'
        })

    # 3. High-Power Appliance Staggering
    if app_usage == 'High' or any(a in appliances_list for a in ['Geyser / Water Heater', 'Washing Machine', 'Electric Oven / Microwave']):
        recs.append({
            'title': 'Stagger Simultaneous High-Wattage Appliances',
            'category': 'Appliance Management',
            'priority': 'High Impact',
            'badge_color': 'warning',
            'icon': 'bolt',
            'description': (
                "Operating high-power heating and motor loads (e.g., water heater, washing machine, electric oven) "
                "at the exact same time creates severe instantaneous peak demand and increases resistive line losses."
            ),
            'action': 'Run heavy appliances sequentially rather than concurrently.'
        })

    # 4. Standby & Phantom Load Management
    if app_usage in ['Medium', 'High']:
        recs.append({
            'title': 'Eliminate Idle Vampire / Standby Loads',
            'category': 'Appliance Management',
            'priority': 'Medium Impact',
            'badge_color': 'info',
            'icon': 'plug',
            'description': (
                "Televisions, set-top boxes, desktop computers, and microwave clocks draw passive vampire power "
                "continuously when left on standby."
            ),
            'action': 'Turn off appliances at the primary wall switch or utilize smart power strips.'
        })

    # 5. Occupancy-Adaptive Lighting and Conditioning
    if occupants_val >= 4:
        recs.append({
            'title': 'Zoned Lighting & Shared Space Utilization',
            'category': 'Behavioral & Lighting',
            'priority': 'Medium Impact',
            'badge_color': 'info',
            'icon': 'users',
            'description': (
                f"With {occupants_val} household occupants, individual room occupancy multiplies lighting and "
                "fan loads across unoccupied zones. Encouraging shared living room usage during peak evening hours "
                "concentrates energy utilization."
            ),
            'action': 'Turn off lights and fans in unoccupied rooms and transition to high-efficiency LEDs.'
        })

    # 6. Low Consumption Reinforcement
    if category == 'Low':
        recs.append({
            'title': 'Maintain Energy-Efficient Operational Baseline',
            'category': 'Efficiency Maintenance',
            'priority': 'Actionable Tip',
            'badge_color': 'success',
            'icon': 'check-circle',
            'description': (
                "Your current household parameters maintain a Low consumption profile. Continue keeping unused "
                "appliances isolated and scheduling preventive maintenance on cooling filters to preserve optimal COP."
            ),
            'action': 'Clean air filters and check refrigerator door gaskets periodically.'
        })

    # Fallback if list is short
    if len(recs) < 3:
        recs.append({
            'title': 'Adopt Smart Thermostat & Inverter Technology',
            'category': 'Long-term Efficiency',
            'priority': 'Actionable Tip',
            'badge_color': 'secondary',
            'icon': 'lightbulb',
            'description': (
                "Inverter-based compressors and 5-star energy rated appliances adapt motor speed to real-time "
                "thermal load rather than constantly cycling fully on and off."
            ),
            'action': 'Evaluate energy star ratings when replacing legacy appliances.'
        })

    disclaimer = (
        "Disclaimer: Decision-support recommendations are advisory guidelines derived from standard energy "
        "conservation heuristics. Actual kilowatt-hour or monetary savings depend on specific appliance wattage, "
        "usage duration, and local utility tariff structures. The system does not claim measured savings without "
        "calibrated hardware sub-metering."
    )

    return recs, disclaimer
