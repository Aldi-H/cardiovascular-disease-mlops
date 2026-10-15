"""Generate the provisioned Grafana dashboard from API metric definitions."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "grafana"
    / "dashboards"
    / "cardiovascular-api-dashboard.json"
)

PANEL_SPECS = [
    {
        "title": "HTTP request rate (requests/sec)",
        "description": "Incoming API requests per second, grouped by route.",
        "expr": "sum by (path) (rate(http_requests_total[5m]))",
        "legend": "{{path}}",
        "unit": "reqps",
    },
    {
        "title": "P95 request latency (seconds)",
        "description": "95th percentile request latency, excluding Prometheus scrapes.",
        "expr": 'histogram_quantile(0.95, sum by (le, path) (rate(http_request_duration_seconds_bucket{path!="/metrics"}[5m])))',
        "legend": "{{path}}",
        "unit": "s",
    },
    {
        "title": "HTTP error rate (responses/sec)",
        "description": "HTTP errors grouped by route and status code.",
        "expr": 'sum by (path, status_code) (rate(http_errors_total{path!="/metrics"}[5m]))',
        "legend": "{{path}} {{status_code}}",
        "unit": "reqps",
    },
    {
        "title": "Prediction rate by class (predictions/sec)",
        "description": "Prediction counts per second; class 1 is a positive prediction.",
        "expr": "sum by (cardio_class) (rate(predictions_total[5m]))",
        "legend": "cardio={{cardio_class}}",
        "unit": "reqps",
    },
    {
        "title": "Mean predicted probability (0–1)",
        "description": "Rolling mean probability for cardio=1.",
        "expr": "rate(prediction_confidence_sum[5m]) / clamp_min(rate(prediction_confidence_count[5m]), 0.001)",
        "legend": "mean probability",
        "unit": "percentunit",
    },
    {
        "title": "Mean input age (years)",
        "description": "Rolling mean input age after conversion from days to years.",
        "expr": "rate(prediction_input_age_years_sum[5m]) / clamp_min(rate(prediction_input_age_years_count[5m]), 0.001)",
        "legend": "age (years)",
        "unit": "short",
    },
    {
        "title": "Mean systolic pressure (mmHg)",
        "description": "Rolling mean systolic pressure supplied to the API.",
        "expr": "rate(prediction_input_systolic_mmhg_sum[5m]) / clamp_min(rate(prediction_input_systolic_mmhg_count[5m]), 0.001)",
        "legend": "systolic (mmHg)",
        "unit": "short",
    },
    {
        "title": "Mean diastolic pressure (mmHg)",
        "description": "Rolling mean diastolic pressure supplied to the API.",
        "expr": "rate(prediction_input_diastolic_mmhg_sum[5m]) / clamp_min(rate(prediction_input_diastolic_mmhg_count[5m]), 0.001)",
        "legend": "diastolic (mmHg)",
        "unit": "short",
    },
    {
        "title": "Mean input height (cm)",
        "description": "Rolling mean height supplied to the API, in centimeters.",
        "expr": "rate(prediction_input_height_cm_sum[5m]) / clamp_min(rate(prediction_input_height_cm_count[5m]), 0.001)",
        "legend": "height (cm)",
        "unit": "short",
    },
    {
        "title": "Mean input weight (kg)",
        "description": "Rolling mean weight supplied to the API, in kilograms.",
        "expr": "rate(prediction_input_weight_kg_sum[5m]) / clamp_min(rate(prediction_input_weight_kg_count[5m]), 0.001)",
        "legend": "weight (kg)",
        "unit": "short",
    },
    {
        "title": "Mean derived BMI",
        "description": "Rolling mean BMI calculated by the serving API.",
        "expr": "rate(prediction_input_bmi_sum[5m]) / clamp_min(rate(prediction_input_bmi_count[5m]), 0.001)",
        "legend": "BMI",
        "unit": "short",
    },
    {
        "title": "Maximum feature drift z-score",
        "description": "Maximum absolute z-score of recent feature means versus training data; drift alert threshold is 2.",
        "expr": "data_drift_max_zscore",
        "legend": "max z-score",
        "unit": "short",
    },
    {
        "title": "Feature drift z-score",
        "description": "Absolute z-score for each numeric feature, based on the latest 50 valid predictions.",
        "expr": "input_feature_drift_zscore",
        "legend": "{{feature}}",
        "unit": "short",
    },
    {
        "title": "Model load duration (seconds)",
        "description": "Time required to load the MLflow model at API startup.",
        "expr": "model_load_duration_seconds",
        "legend": "model load",
        "unit": "s",
    },
    {
        "title": "Requests in flight (requests)",
        "description": "Currently processed requests; the /metrics scrape is excluded.",
        "expr": "http_requests_in_flight",
        "legend": "in flight",
        "unit": "short",
    },
    {
        "title": "Application uptime (seconds)",
        "description": "Time since the FastAPI process started.",
        "expr": "application_uptime_seconds",
        "legend": "uptime",
        "unit": "s",
    },
    {
        "title": "Process CPU use (cores)",
        "description": "CPU seconds consumed per second by the API process.",
        "expr": "rate(process_cpu_seconds_total[5m])",
        "legend": "CPU cores",
        "unit": "short",
    },
    {
        "title": "Process resident memory (bytes)",
        "description": "Resident memory used by the API process.",
        "expr": "process_resident_memory_bytes",
        "legend": "resident memory",
        "unit": "bytes",
    },
]


def make_panel(panel_id: int, spec: dict) -> dict:
    row = (panel_id - 1) // 2
    column = (panel_id - 1) % 2
    return {
        "id": panel_id,
        "type": "timeseries",
        "title": spec["title"],
        "description": spec["description"],
        "datasource": {"type": "prometheus", "uid": "prometheus"},
        "gridPos": {"h": 8, "w": 12, "x": column * 12, "y": row * 8},
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
                "custom": {
                    "drawStyle": "line",
                    "fillOpacity": 10,
                    "lineWidth": 2,
                    "showPoints": "never",
                },
                "unit": spec["unit"],
            },
            "overrides": [],
        },
        "options": {
            "legend": {
                "calcs": [],
                "displayMode": "list",
                "placement": "bottom",
                "showLegend": True,
            },
            "tooltip": {"mode": "single", "sort": "none"},
        },
        "targets": [
            {
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "editorMode": "code",
                "expr": spec["expr"],
                "legendFormat": spec["legend"],
                "range": True,
                "refId": "A",
            }
        ],
    }


def main() -> None:
    dashboard = {
        "annotations": {
            "list": [
                {
                    "builtIn": 1,
                    "datasource": {"type": "grafana", "uid": "-- Grafana --"},
                    "enable": True,
                    "hide": True,
                    "iconColor": "rgba(0, 211, 255, 1)",
                    "name": "Annotations & Alerts",
                    "type": "dashboard",
                }
            ]
        },
        "editable": True,
        "graphTooltip": 0,
        "panels": [
            make_panel(panel_id, spec)
            for panel_id, spec in enumerate(PANEL_SPECS, start=1)
        ],
        "refresh": "15s",
        "schemaVersion": 39,
        "tags": ["cardiovascular", "prometheus", "mlops"],
        "templating": {"list": []},
        "time": {"from": "now-1h", "to": "now"},
        "timezone": "browser",
        "title": "Cardiovascular Disease API - Aldi Himawan",
        "uid": "cardiovascular-api",
        "version": 1,
    }

    dashboard_path = (
        PROJECT_ROOT
        / "monitoring"
        / "grafana"
        / "dashboards"
        / "cardiovascular-api-dashboard.json"
    )
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_path.write_text(
        json.dumps(dashboard, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Dashboard generated: {dashboard_path}")
    print(f"Panels: {len(PANEL_SPECS)}")


if __name__ == "__main__":
    main()
