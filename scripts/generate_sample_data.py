"""
============================================================================
SGT400 Gas Turbine - Sample Dataset Generator
============================================================================
Generates realistic time-series sensor data for a Siemens SGT400 gas turbine.
Includes normal operation, degradation patterns, and fault injection.

Sensors modeled:
- Exhaust Gas Temperature (EGT)
- Compressor Discharge Temperature (CDT)
- Vibration (bearing housing)
- Compressor Inlet Pressure
- Compressor Discharge Pressure
- Turbine Load (MW)
- Fuel Flow Rate
- Rotor Speed (RPM)
- Lube Oil Temperature
- Ambient Temperature

Reference: https://learn.microsoft.com/fabric/data-engineering/lakehouse-overview
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------- Turbine Baseline Parameters (SGT400 ~13MW) ----------
TURBINE_PARAMS = {
    "exhaust_gas_temp_c": {"mean": 545, "std": 8, "min": 480, "max": 620, "unit": "°C"},
    "compressor_discharge_temp_c": {"mean": 380, "std": 5, "min": 340, "max": 420, "unit": "°C"},
    "vibration_mm_s": {"mean": 2.5, "std": 0.4, "min": 0.5, "max": 12.0, "unit": "mm/s"},
    "inlet_pressure_bar": {"mean": 1.013, "std": 0.01, "min": 0.95, "max": 1.08, "unit": "bar"},
    "discharge_pressure_bar": {"mean": 15.5, "std": 0.3, "min": 13.0, "max": 17.5, "unit": "bar"},
    "turbine_load_mw": {"mean": 11.8, "std": 0.5, "min": 4.0, "max": 13.2, "unit": "MW"},
    "fuel_flow_kg_s": {"mean": 3.2, "std": 0.15, "min": 1.0, "max": 4.5, "unit": "kg/s"},
    "rotor_speed_rpm": {"mean": 9500, "std": 30, "min": 8000, "max": 10200, "unit": "RPM"},
    "lube_oil_temp_c": {"mean": 52, "std": 2, "min": 35, "max": 75, "unit": "°C"},
    "ambient_temp_c": {"mean": 28, "std": 4, "min": 10, "max": 48, "unit": "°C"},
}

# ---------- Fault Profiles ----------
FAULT_PROFILES = {
    "compressor_fouling": {
        "description": "Gradual compressor fouling - reduces efficiency over weeks",
        "affected_sensors": {
            "discharge_pressure_bar": {"drift": -0.05, "noise_mult": 1.3},
            "exhaust_gas_temp_c": {"drift": 0.3, "noise_mult": 1.2},
            "fuel_flow_kg_s": {"drift": 0.02, "noise_mult": 1.1},
            "turbine_load_mw": {"drift": -0.01, "noise_mult": 1.0},
        },
        "ramp_days": 30,
    },
    "bearing_degradation": {
        "description": "Bearing wear causing increased vibration",
        "affected_sensors": {
            "vibration_mm_s": {"drift": 0.15, "noise_mult": 2.0},
            "lube_oil_temp_c": {"drift": 0.1, "noise_mult": 1.5},
            "rotor_speed_rpm": {"drift": -0.5, "noise_mult": 1.3},
        },
        "ramp_days": 14,
    },
    "combustion_instability": {
        "description": "Flame instability causing temperature spikes",
        "affected_sensors": {
            "exhaust_gas_temp_c": {"drift": 0.8, "noise_mult": 2.5},
            "compressor_discharge_temp_c": {"drift": 0.3, "noise_mult": 1.8},
            "fuel_flow_kg_s": {"drift": 0.05, "noise_mult": 2.0},
            "vibration_mm_s": {"drift": 0.05, "noise_mult": 1.5},
        },
        "ramp_days": 7,
    },
    "hot_gas_path_erosion": {
        "description": "Turbine blade erosion from hot gas corrosion",
        "affected_sensors": {
            "exhaust_gas_temp_c": {"drift": 0.5, "noise_mult": 1.4},
            "turbine_load_mw": {"drift": -0.03, "noise_mult": 1.2},
            "rotor_speed_rpm": {"drift": -1.0, "noise_mult": 1.1},
            "vibration_mm_s": {"drift": 0.08, "noise_mult": 1.6},
        },
        "ramp_days": 45,
    },
}

ALARM_CODES = {
    "ALM001": {"description": "High Exhaust Gas Temperature", "severity": "CRITICAL"},
    "ALM002": {"description": "High Vibration Level", "severity": "WARNING"},
    "ALM003": {"description": "Low Lube Oil Pressure", "severity": "CRITICAL"},
    "ALM004": {"description": "Compressor Surge Detected", "severity": "CRITICAL"},
    "ALM005": {"description": "Fuel Flow Deviation", "severity": "WARNING"},
    "ALM006": {"description": "Bearing Temperature High", "severity": "WARNING"},
    "ALM007": {"description": "Rotor Overspeed Warning", "severity": "CRITICAL"},
    "ALM008": {"description": "Flame Instability", "severity": "WARNING"},
    "ALM009": {"description": "Discharge Pressure Low", "severity": "INFO"},
    "ALM010": {"description": "Scheduled Maintenance Due", "severity": "INFO"},
}


def generate_diurnal_pattern(hours: np.ndarray) -> np.ndarray:
    """Simulate daily temperature/load cycle."""
    return np.sin(2 * np.pi * hours / 24 - np.pi / 2) * 0.5 + 0.5


def generate_sensor_data(
    num_days: int = 90,
    interval_minutes: int = 5,
    fault_type: str | None = None,
    fault_start_day: int | None = None,
    turbine_id: str = "SGT400-001",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic SGT400 turbine sensor time-series data.
    
    Args:
        num_days: Number of days of data to generate
        interval_minutes: Sampling interval in minutes
        fault_type: Optional fault profile to inject
        fault_start_day: Day when fault begins (if fault_type specified)
        turbine_id: Turbine identifier
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with timestamp-indexed sensor readings
    """
    np.random.seed(seed)
    
    samples_per_day = (24 * 60) // interval_minutes
    total_samples = num_days * samples_per_day
    
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=i * interval_minutes) for i in range(total_samples)]
    hours = np.array([(t.hour + t.minute / 60) for t in timestamps])
    
    diurnal = generate_diurnal_pattern(hours)
    
    data = {"timestamp": timestamps, "turbine_id": [turbine_id] * total_samples}
    
    # Generate base sensor readings
    for sensor, params in TURBINE_PARAMS.items():
        base = params["mean"]
        noise = np.random.normal(0, params["std"], total_samples)
        
        # Add diurnal variation for temperature sensors
        if "temp" in sensor:
            diurnal_effect = diurnal * params["std"] * 2
            signal = base + noise + diurnal_effect
        elif sensor == "turbine_load_mw":
            # Load follows demand pattern
            load_pattern = 0.7 + 0.3 * diurnal
            signal = base * load_pattern + noise * 0.5
        else:
            signal = base + noise
        
        # Add slow random walk for realism
        random_walk = np.cumsum(np.random.normal(0, params["std"] * 0.01, total_samples))
        random_walk -= np.mean(random_walk)  # Center it
        signal += random_walk
        
        data[sensor] = np.clip(signal, params["min"], params["max"])
    
    df = pd.DataFrame(data)
    
    # Inject fault if specified
    if fault_type and fault_type in FAULT_PROFILES:
        fault_start_day = fault_start_day or (num_days // 2)
        fault_profile = FAULT_PROFILES[fault_type]
        fault_start_idx = fault_start_day * samples_per_day
        ramp_samples = fault_profile["ramp_days"] * samples_per_day
        
        logger.info(f"Injecting fault '{fault_type}' starting day {fault_start_day}")
        
        for sensor, effect in fault_profile["affected_sensors"].items():
            for i in range(fault_start_idx, total_samples):
                progress = min(1.0, (i - fault_start_idx) / ramp_samples)
                # Exponential ramp for realistic degradation
                severity = progress ** 1.5
                
                drift = effect["drift"] * severity * TURBINE_PARAMS[sensor]["std"]
                noise_mult = 1.0 + (effect["noise_mult"] - 1.0) * severity
                
                additional_noise = np.random.normal(0, TURBINE_PARAMS[sensor]["std"] * (noise_mult - 1))
                df.loc[i, sensor] += drift * (i - fault_start_idx) / samples_per_day + additional_noise
                df.loc[i, sensor] = np.clip(
                    df.loc[i, sensor],
                    TURBINE_PARAMS[sensor]["min"],
                    TURBINE_PARAMS[sensor]["max"],
                )
        
        # Add fault labels
        df["fault_type"] = "normal"
        df.loc[fault_start_idx:, "fault_type"] = fault_type
        df["days_to_failure"] = np.nan
        remaining_days = num_days - fault_start_day
        df.loc[fault_start_idx:, "days_to_failure"] = np.linspace(
            remaining_days, 0, total_samples - fault_start_idx
        )
    else:
        df["fault_type"] = "normal"
        df["days_to_failure"] = np.nan
    
    # Compute derived features
    df["pressure_ratio"] = df["discharge_pressure_bar"] / df["inlet_pressure_bar"]
    df["heat_rate"] = df["fuel_flow_kg_s"] * 3600 * 45 / (df["turbine_load_mw"] + 0.01)  # MJ/MWh approx
    df["efficiency_pct"] = np.clip(
        (df["turbine_load_mw"] / (df["fuel_flow_kg_s"] * 45 + 0.01)) * 100 / 0.38, 50, 110
    )
    
    return df


def generate_alarm_logs(
    sensor_df: pd.DataFrame,
    turbine_id: str = "SGT400-001",
) -> pd.DataFrame:
    """Generate alarm logs based on sensor thresholds."""
    alarms = []
    
    thresholds = {
        "ALM001": ("exhaust_gas_temp_c", ">", 580),
        "ALM002": ("vibration_mm_s", ">", 5.0),
        "ALM005": ("fuel_flow_kg_s", ">", 4.0),
        "ALM006": ("lube_oil_temp_c", ">", 65),
        "ALM009": ("discharge_pressure_bar", "<", 14.0),
    }
    
    for alarm_code, (sensor, op, threshold) in thresholds.items():
        if op == ">":
            mask = sensor_df[sensor] > threshold
        else:
            mask = sensor_df[sensor] < threshold
        
        triggered = sensor_df[mask]
        for _, row in triggered.iterrows():
            alarms.append({
                "timestamp": row["timestamp"],
                "turbine_id": turbine_id,
                "alarm_code": alarm_code,
                "alarm_description": ALARM_CODES[alarm_code]["description"],
                "severity": ALARM_CODES[alarm_code]["severity"],
                "sensor_value": row[sensor],
                "threshold": threshold,
                "acknowledged": np.random.random() > 0.3,
            })
    
    if alarms:
        return pd.DataFrame(alarms).sort_values("timestamp").reset_index(drop=True)
    return pd.DataFrame(columns=[
        "timestamp", "turbine_id", "alarm_code", "alarm_description",
        "severity", "sensor_value", "threshold", "acknowledged",
    ])


def generate_maintenance_log(num_days: int, turbine_id: str = "SGT400-001") -> pd.DataFrame:
    """Generate historical maintenance records."""
    records = []
    start = datetime(2025, 1, 1)
    
    maintenance_types = [
        {"type": "Scheduled Inspection", "interval_days": 30, "duration_hours": 4},
        {"type": "Oil Change", "interval_days": 90, "duration_hours": 2},
        {"type": "Filter Replacement", "interval_days": 60, "duration_hours": 3},
        {"type": "Vibration Analysis", "interval_days": 14, "duration_hours": 1},
        {"type": "Borescope Inspection", "interval_days": 180, "duration_hours": 8},
    ]
    
    for maint in maintenance_types:
        day = maint["interval_days"]
        while day < num_days:
            records.append({
                "timestamp": start + timedelta(days=day, hours=np.random.randint(6, 18)),
                "turbine_id": turbine_id,
                "maintenance_type": maint["type"],
                "duration_hours": maint["duration_hours"] + np.random.uniform(-0.5, 1.0),
                "cost_usd": np.random.uniform(500, 15000),
                "technician": f"TECH-{np.random.randint(100, 999)}",
                "notes": f"Routine {maint['type'].lower()} completed successfully",
            })
            day += maint["interval_days"]
    
    return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)


def save_datasets(output_dir: str = "data/sample", num_days: int = 90) -> dict:
    """
    Generate and save all datasets.
    
    Returns:
        Dictionary with file paths and row counts
    """
    os.makedirs(output_dir, exist_ok=True)
    summary = {}
    
    # 1. Normal operation data
    logger.info("Generating normal operation data...")
    df_normal = generate_sensor_data(num_days=num_days, fault_type=None, seed=42)
    path = os.path.join(output_dir, "sensor_data_normal.parquet")
    df_normal.to_parquet(path, index=False, engine="pyarrow")
    summary["sensor_data_normal"] = {"path": path, "rows": len(df_normal)}
    
    # Also save CSV sample (first 1000 rows)
    csv_path = os.path.join(output_dir, "sensor_data_normal_sample.csv")
    df_normal.head(1000).to_csv(csv_path, index=False)
    
    # 2. Fault scenarios
    for fault_name in FAULT_PROFILES:
        logger.info(f"Generating {fault_name} fault data...")
        df_fault = generate_sensor_data(
            num_days=num_days,
            fault_type=fault_name,
            fault_start_day=num_days // 3,
            seed=hash(fault_name) % (2**31),
        )
        path = os.path.join(output_dir, f"sensor_data_{fault_name}.parquet")
        df_fault.to_parquet(path, index=False, engine="pyarrow")
        summary[f"sensor_data_{fault_name}"] = {"path": path, "rows": len(df_fault)}
    
    # 3. Combined training dataset (all scenarios)
    logger.info("Generating combined training dataset...")
    all_dfs = [df_normal]
    for i, fault_name in enumerate(FAULT_PROFILES):
        df_f = generate_sensor_data(
            num_days=num_days,
            fault_type=fault_name,
            fault_start_day=num_days // 3,
            turbine_id=f"SGT400-{i+2:03d}",
            seed=hash(fault_name) % (2**31) + 100,
        )
        all_dfs.append(df_f)
    
    df_combined = pd.concat(all_dfs, ignore_index=True)
    path = os.path.join(output_dir, "sensor_data_training.parquet")
    df_combined.to_parquet(path, index=False, engine="pyarrow")
    summary["sensor_data_training"] = {"path": path, "rows": len(df_combined)}
    
    # 4. Alarm logs
    logger.info("Generating alarm logs...")
    df_alarms = generate_alarm_logs(df_combined)
    path = os.path.join(output_dir, "alarm_logs.parquet")
    df_alarms.to_parquet(path, index=False, engine="pyarrow")
    summary["alarm_logs"] = {"path": path, "rows": len(df_alarms)}
    
    # 5. Maintenance log
    logger.info("Generating maintenance log...")
    df_maintenance = generate_maintenance_log(num_days)
    path = os.path.join(output_dir, "maintenance_log.parquet")
    df_maintenance.to_parquet(path, index=False, engine="pyarrow")
    summary["maintenance_log"] = {"path": path, "rows": len(df_maintenance)}
    
    # Save summary
    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"All datasets saved to {output_dir}")
    for name, info in summary.items():
        logger.info(f"  {name}: {info['rows']:,} rows -> {info['path']}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SGT400 turbine sample data")
    parser.add_argument("--output-dir", default="data/sample", help="Output directory")
    parser.add_argument("--num-days", type=int, default=90, help="Number of days")
    parser.add_argument("--interval", type=int, default=5, help="Sampling interval (minutes)")
    args = parser.parse_args()
    
    save_datasets(output_dir=args.output_dir, num_days=args.num_days)
