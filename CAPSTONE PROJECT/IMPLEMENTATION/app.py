"""
Flask Web Application for Smart Home Energy Consumption Prediction
==================================================================
Bayesian Network-Based Smart Home Energy Consumption Prediction
and Decision Support System.

Intelligent Decision Support Platform
"""

import io
import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, Response

import bayesian_model
import database
import recommendations
import charts

app = Flask(__name__)
app.secret_key = 'smart-home-bayesian-energy-secret-key-2026'

# Ensure database is initialized on startup
database.init_db()


@app.route('/', methods=['GET'])
def index():
    """Renders the main dashboard and prediction input form."""
    # Check if a recent prediction result is stored in session or pass defaults
    return render_template(
        'index.html',
        clean_inputs=None,
        prediction_result=None,
        recs=None,
        recs_disclaimer=None
    )


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles form submission:
    1. Validates & preprocesses inputs (Module 1)
    2. Runs Bayesian Network inference (Module 2)
    3. Generates decision support recommendations (Module 3)
    4. Saves scenario to SQLite database (Module 4)
    5. Re-generates Matplotlib analytical charts
    """
    try:
        temp_val = request.form.get('temperature', 28.0)
        occ_val = request.form.get('occupants', 3)
        ac_val = request.form.get('ac_usage', 'Moderate')
        app_cat_val = request.form.get('appliance_usage', 'Medium')
        tod_val = request.form.get('time_of_day', 'Evening')
        day_val = request.form.get('day_type', 'Weekday')
        selected_appliances = request.form.getlist('appliances')

        # Module 1: Preprocessing
        clean_inputs, evidence = bayesian_model.preprocess_household_data(
            temperature_raw=temp_val,
            occupants_raw=occ_val,
            ac_usage_raw=ac_val,
            appliance_usage_raw=app_cat_val,
            time_of_day_raw=tod_val,
            day_type_raw=day_val,
            appliances_list=selected_appliances
        )

        # Module 2: Bayesian Network Inference
        prediction_result = bayesian_model.predict_energy_consumption(evidence)

        # Module 3: Decision Support Recommendations
        recs, recs_disclaimer = recommendations.generate_recommendations(
            clean_inputs, prediction_result
        )

        # Module 4: Database Storage
        database.insert_prediction(clean_inputs, prediction_result)

        # Update analytical charts in background
        df = database.export_dataframe()
        charts.generate_all_charts(df)

        return render_template(
            'index.html',
            clean_inputs=clean_inputs,
            evidence=evidence,
            prediction_result=prediction_result,
            recs=recs,
            recs_disclaimer=recs_disclaimer,
            just_predicted=True
        )

    except Exception as e:
        flash(f"Prediction Error: {str(e)}", "danger")
        return redirect(url_for('index'))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for programmatic inference or asynchronous frontend requests."""
    try:
        data = request.get_json(force=True) if request.is_json else request.form
        clean_inputs, evidence = bayesian_model.preprocess_household_data(
            temperature_raw=data.get('temperature', 28.0),
            occupants_raw=data.get('occupants', 3),
            ac_usage_raw=data.get('ac_usage', 'Moderate'),
            appliance_usage_raw=data.get('appliance_usage', 'Medium'),
            time_of_day_raw=data.get('time_of_day', 'Evening'),
            day_type_raw=data.get('day_type', 'Weekday'),
            appliances_list=data.get('appliances', [])
        )

        prediction_result = bayesian_model.predict_energy_consumption(evidence)
        recs, disclaimer = recommendations.generate_recommendations(clean_inputs, prediction_result)
        pred_id = database.insert_prediction(clean_inputs, prediction_result)

        return jsonify({
            'status': 'success',
            'prediction_id': pred_id,
            'clean_inputs': clean_inputs,
            'evidence': evidence,
            'prediction_result': prediction_result,
            'recommendations': recs,
            'disclaimer': disclaimer
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/history', methods=['GET'])
def history():
    """Module 4: Displays historical prediction table and options."""
    all_predictions = database.get_all_predictions()
    stats = database.get_summary_stats()
    return render_template(
        'history.html',
        predictions=all_predictions,
        stats=stats
    )


@app.route('/history/delete/<int:pred_id>', methods=['POST'])
def delete_history_item(pred_id):
    """Deletes an individual prediction record."""
    database.delete_prediction(pred_id)
    flash(f"Prediction record #{pred_id} successfully deleted.", "info")
    # Refresh charts
    df = database.export_dataframe()
    charts.generate_all_charts(df)
    return redirect(url_for('history'))


@app.route('/history/clear', methods=['POST'])
def clear_history():
    """Clears all prediction records."""
    database.clear_all_predictions()
    flash("Prediction history cleared.", "warning")
    df = database.export_dataframe()
    charts.generate_all_charts(df)
    return redirect(url_for('history'))


@app.route('/history/seed', methods=['POST'])
def seed_history():
    """Seeds sample benchmark scenarios for system demonstration."""
    database.seed_benchmark_samples()
    df = database.export_dataframe()
    charts.generate_all_charts(df)
    flash("Sample benchmark scenarios loaded successfully for project demonstration.", "success")
    return redirect(url_for('history'))


@app.route('/analytics', methods=['GET'])
def analytics():
    """Module 4: Displays graphical charts, KPIs, daily, and monthly summaries."""
    stats = database.get_summary_stats()
    daily_summary = database.get_daily_summary()
    monthly_summary = database.get_monthly_summary()

    # Generate charts if database has entries
    df = database.export_dataframe()
    chart_paths = charts.generate_all_charts(df)

    return render_template(
        'analytics.html',
        stats=stats,
        daily_summary=daily_summary,
        monthly_summary=monthly_summary,
        chart_paths=chart_paths,
        total_records=len(df)
    )


@app.route('/export/csv', methods=['GET'])
def export_csv():
    """Exports prediction history as a downloadable CSV report."""
    df = database.export_dataframe()
    if df.empty:
        flash("No prediction history available to export.", "info")
        return redirect(url_for('history'))

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_output = csv_buffer.getvalue()

    return Response(
        csv_output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=smart_home_energy_predictions.csv"}
    )


@app.route('/model-info', methods=['GET'])
def model_info():
    """Information page explaining the Bayesian Network structure and theory."""
    return render_template('model_info.html')


if __name__ == '__main__':
    # Default to localhost port 5000 for local development
    port = int(os.environ.get('PORT', 5000))
    print(f"\n=======================================================")
    print(f" Smart Home Energy Prediction & Decision Support System")
    print(f" Running locally at: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host='127.0.0.1', port=port, debug=True)
