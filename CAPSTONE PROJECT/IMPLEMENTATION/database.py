"""
Database Access Layer for Smart Home Energy Prediction History
==============================================================
Module 4: SQLite Database management for storing household prediction logs,
calculating summary aggregates, and exporting records.
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime

# Place database file in the same directory as this script
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'energy_predictions.db')


def get_db_connection():
    """Returns a SQLite database connection with row factory enabled."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the SQLite schema automatically if it does not already exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            temperature_val REAL NOT NULL,
            temperature_cat TEXT NOT NULL,
            occupants_val INTEGER NOT NULL,
            occupants_cat TEXT NOT NULL,
            ac_usage TEXT NOT NULL,
            appliance_usage TEXT NOT NULL,
            time_of_day TEXT NOT NULL,
            day_type TEXT NOT NULL,
            appliances_list TEXT,
            predicted_category TEXT NOT NULL,
            prob_low REAL NOT NULL,
            prob_medium REAL NOT NULL,
            prob_high REAL NOT NULL,
            explanation TEXT,
            is_benchmark INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


def insert_prediction(clean_inputs, prediction_result, is_benchmark=0):
    """
    Inserts a newly computed prediction into the predictions table.
    Returns:
        inserted_id: The primary key of the new record.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    appliances_str = ", ".join(clean_inputs.get('appliances_list', []))
    probs = prediction_result.get('probabilities', {})

    cursor.execute("""
        INSERT INTO predictions (
            timestamp,
            temperature_val,
            temperature_cat,
            occupants_val,
            occupants_cat,
            ac_usage,
            appliance_usage,
            time_of_day,
            day_type,
            appliances_list,
            predicted_category,
            prob_low,
            prob_medium,
            prob_high,
            explanation,
            is_benchmark
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        timestamp_str,
        clean_inputs.get('temperature_val', 26.0),
        clean_inputs.get('temperature_cat', 'Moderate'),
        clean_inputs.get('occupants_val', 3),
        clean_inputs.get('occupants_cat', 'Medium'),
        clean_inputs.get('ac_usage', 'Moderate'),
        clean_inputs.get('appliance_usage', 'Medium'),
        clean_inputs.get('time_of_day', 'Evening'),
        clean_inputs.get('day_type', 'Weekday'),
        appliances_str,
        prediction_result.get('predicted_category', 'Medium'),
        probs.get('Low', 0.0),
        probs.get('Medium', 0.0),
        probs.get('High', 0.0),
        prediction_result.get('explanation', ''),
        is_benchmark
    ))

    inserted_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return inserted_id


def get_all_predictions(limit=200):
    """Retrieves prediction records ordered by latest timestamp first."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM predictions
        ORDER BY id DESC
        LIMIT ?;
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_prediction_by_id(pred_id):
    """Fetches a single prediction by its ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE id = ?;", (pred_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_prediction(pred_id):
    """Deletes a specific prediction entry."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE id = ?;", (pred_id,))
    conn.commit()
    conn.close()


def clear_all_predictions():
    """Clears all records from the predictions table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions;")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='predictions';")
    conn.commit()
    conn.close()


def get_summary_stats():
    """Computes key metrics from recorded prediction logs."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Total count
    cursor.execute("SELECT COUNT(*) AS total FROM predictions;")
    total = cursor.fetchone()['total']

    if total == 0:
        conn.close()
        return {
            'total': 0,
            'count_low': 0,
            'count_medium': 0,
            'count_high': 0,
            'pct_low': 0.0,
            'pct_medium': 0.0,
            'pct_high': 0.0,
            'dominant_category': 'None',
            'dominant_tod': 'None',
            'today_count': 0,
            'month_count': 0
        }

    # Count by category
    cursor.execute("""
        SELECT predicted_category, COUNT(*) AS count
        FROM predictions
        GROUP BY predicted_category;
    """)
    cat_counts = {r['predicted_category']: r['count'] for r in cursor.fetchall()}
    c_low = cat_counts.get('Low', 0)
    c_med = cat_counts.get('Medium', 0)
    c_high = cat_counts.get('High', 0)

    # Dominant category
    dominant_cat = max(cat_counts, key=cat_counts.get) if cat_counts else 'None'

    # Dominant Time of Day
    cursor.execute("""
        SELECT time_of_day, COUNT(*) AS count
        FROM predictions
        GROUP BY time_of_day
        ORDER BY count DESC
        LIMIT 1;
    """)
    tod_row = cursor.fetchone()
    dominant_tod = tod_row['time_of_day'] if tod_row else 'None'

    # Today's predictions
    today_str = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("SELECT COUNT(*) AS today_cnt FROM predictions WHERE timestamp LIKE ?;", (f"{today_str}%",))
    today_cnt = cursor.fetchone()['today_cnt']

    # Month's predictions
    month_str = datetime.now().strftime("%Y-%m")
    cursor.execute("SELECT COUNT(*) AS month_cnt FROM predictions WHERE timestamp LIKE ?;", (f"{month_str}%",))
    month_cnt = cursor.fetchone()['month_cnt']

    conn.close()

    return {
        'total': total,
        'count_low': c_low,
        'count_medium': c_med,
        'count_high': c_high,
        'pct_low': round((c_low / total) * 100, 1) if total > 0 else 0,
        'pct_medium': round((c_med / total) * 100, 1) if total > 0 else 0,
        'pct_high': round((c_high / total) * 100, 1) if total > 0 else 0,
        'dominant_category': dominant_cat,
        'dominant_tod': dominant_tod,
        'today_count': today_cnt,
        'month_count': month_cnt
    }


def get_daily_summary():
    """Generates daily aggregated counts for predictions table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            substr(timestamp, 1, 10) AS pred_date,
            SUM(CASE WHEN predicted_category = 'Low' THEN 1 ELSE 0 END) AS low_count,
            SUM(CASE WHEN predicted_category = 'Medium' THEN 1 ELSE 0 END) AS medium_count,
            SUM(CASE WHEN predicted_category = 'High' THEN 1 ELSE 0 END) AS high_count,
            COUNT(*) AS total_count
        FROM predictions
        GROUP BY pred_date
        ORDER BY pred_date DESC
        LIMIT 30;
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_monthly_summary():
    """Generates monthly aggregated counts for predictions table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            substr(timestamp, 1, 7) AS pred_month,
            SUM(CASE WHEN predicted_category = 'Low' THEN 1 ELSE 0 END) AS low_count,
            SUM(CASE WHEN predicted_category = 'Medium' THEN 1 ELSE 0 END) AS medium_count,
            SUM(CASE WHEN predicted_category = 'High' THEN 1 ELSE 0 END) AS high_count,
            COUNT(*) AS total_count
        FROM predictions
        GROUP BY pred_month
        ORDER BY pred_month DESC
        LIMIT 12;
    """)
    rows = cursor.fetchall()
    conn.close()

    result = []
    for r in rows:
        d = dict(r)
        d['high_pct'] = round((d['high_count'] / d['total_count']) * 100.0, 1) if d['total_count'] > 0 else 0.0
        result.append(d)
    return result


def export_dataframe():
    """Returns all predictions as a pandas DataFrame for CSV export."""
    conn = get_db_connection()
    df = pd.read_sql_query("""
        SELECT 
            id AS "Prediction_ID",
            timestamp AS "Date_Time",
            temperature_val AS "Temperature_C",
            temperature_cat AS "Temperature_Class",
            occupants_val AS "Occupants_Count",
            occupants_cat AS "Occupants_Class",
            ac_usage AS "AC_Usage",
            appliance_usage AS "Appliance_Usage",
            time_of_day AS "Time_Of_Day",
            day_type AS "Day_Type",
            appliances_list AS "Appliances_Selected",
            predicted_category AS "Predicted_Category",
            prob_low AS "Prob_Low_Percent",
            prob_medium AS "Prob_Medium_Percent",
            prob_high AS "Prob_High_Percent",
            is_benchmark AS "Is_Benchmark_Sample"
        FROM predictions
        ORDER BY id ASC;
    """, conn)
    conn.close()
    return df


def seed_benchmark_samples():
    """
    Populates realistic benchmark demonstration scenarios.
    Useful for demonstration and evaluation so charts are immediately populated.
    All entries are explicitly flagged with is_benchmark=1.
    """
    benchmark_cases = [
        # Date offset scenarios for recent days
        {"temp": 36.5, "occ": 5, "ac": "Heavy", "app": "High", "tod": "Evening", "day": "Weekend", "apps": ["Geyser / Water Heater", "Washing Machine", "Electric Oven / Microwave"], "cat": "High", "p": (2.0, 18.0, 80.0), "exp": "Heavy AC in peak evening with high occupancy drives High consumption.", "date": "2026-10-14 19:30:00"},
        {"temp": 34.0, "occ": 4, "ac": "Heavy", "app": "Medium", "tod": "Afternoon", "day": "Weekday", "apps": ["Refrigerator", "Desktop PC / Monitor"], "cat": "High", "p": (5.0, 30.0, 65.0), "exp": "Hot afternoon combined with continuous AC cooling demands High power.", "date": "2026-10-14 14:15:00"},
        {"temp": 28.0, "occ": 3, "ac": "Moderate", "app": "Medium", "tod": "Morning", "day": "Weekday", "apps": ["Refrigerator", "Television / Entertainment"], "cat": "Medium", "p": (25.0, 55.0, 20.0), "exp": "Moderate temperature and standard morning routine maintain Medium consumption.", "date": "2026-10-15 08:45:00"},
        {"temp": 19.5, "occ": 2, "ac": "Off", "app": "Low", "tod": "Night", "day": "Weekday", "apps": ["Refrigerator"], "cat": "Low", "p": (85.0, 12.0, 3.0), "exp": "Nighttime sleep and zero AC cooling maintain Low baseline consumption.", "date": "2026-10-15 23:20:00"},
        {"temp": 32.0, "occ": 4, "ac": "Moderate", "app": "High", "tod": "Evening", "day": "Weekend", "apps": ["Washing Machine", "Dishwasher", "Television / Entertainment"], "cat": "High", "p": (8.0, 32.0, 60.0), "exp": "Concurrent laundry, dishwasher, and evening leisure push consumption to High.", "date": "2026-10-16 20:10:00"},
        {"temp": 25.0, "occ": 2, "ac": "Off", "app": "Low", "tod": "Morning", "day": "Weekend", "apps": ["Refrigerator", "Lighting / Fans"], "cat": "Low", "p": (78.0, 18.0, 4.0), "exp": "Mild morning temperature with AC off ensures Low energy consumption.", "date": "2026-10-16 09:30:00"},
        {"temp": 27.0, "occ": 3, "ac": "Moderate", "app": "Medium", "tod": "Afternoon", "day": "Weekend", "apps": ["Refrigerator", "Television / Entertainment", "Desktop PC / Monitor"], "cat": "Medium", "p": (20.0, 62.0, 18.0), "exp": "Moderate cooling and standard entertainment electronics sustain Medium consumption.", "date": "2026-10-17 10:45:00"}
    ]

    conn = get_db_connection()
    cursor = conn.cursor()

    for item in benchmark_cases:
        t_cat = 'Hot' if item['temp'] > 30 else ('Cool' if item['temp'] < 22 else 'Moderate')
        o_cat = 'High' if item['occ'] >= 5 else ('Medium' if item['occ'] >= 3 else 'Low')
        apps_str = ", ".join(item['apps'])
        cursor.execute("""
            INSERT INTO predictions (
                timestamp, temperature_val, temperature_cat, occupants_val, occupants_cat,
                ac_usage, appliance_usage, time_of_day, day_type, appliances_list,
                predicted_category, prob_low, prob_medium, prob_high, explanation, is_benchmark
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1);
        """, (
            item['date'], item['temp'], t_cat, item['occ'], o_cat,
            item['ac'], item['app'], item['tod'], item['day'], apps_str,
            item['cat'], item['p'][0], item['p'][1], item['p'][2], item['exp']
        ))

    conn.commit()
    conn.close()
