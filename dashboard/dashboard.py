"""
dashboard/dashboard.py
Interactive Plotly Dash dashboard for churn analysis and live prediction.

Run:
  python dashboard/dashboard.py
  Open http://localhost:8050
"""

import json
import os
import pickle
import sys
from urllib.parse import quote

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "telecom_churn.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


df_raw = pd.read_csv(DATA_PATH)


def _load(fname):
    with open(os.path.join(MODELS_DIR, fname), "rb") as f:
        return pickle.load(f)



def _icon_data_uri(svg_markup):
    return f"data:image/svg+xml;utf8,{quote(svg_markup)}"


try:
    MODEL = _load("best_model.pkl")
    SCALER = _load("scaler.pkl")
    ENCODERS = _load("encoders.pkl")
    FEATURE_NAMES = _load("features.pkl")
    with open(os.path.join(MODELS_DIR, "meta.json"), encoding="utf-8") as f:
        META = json.load(f)
    MODEL_LOADED = True
except Exception:
    MODEL_LOADED = False
    META = {"model_name": "N/A", "roc_auc": "N/A"}


COLORS = {
    "navy": "#16324f",
    "slate": "#486581",
    "sky": "#4f8cff",
    "cyan": "#4cc9f0",
    "mint": "#67d5b5",
    "gold": "#f4b860",
    "coral": "#f76e6e",
    "surface": "#f6f7fb",
    "panel": "#ffffff",
    "text": "#1f2937",
    "muted": "#6b7280",
    "grid": "#dbe4f0",
}

BLUE = COLORS["sky"]
RED = COLORS["coral"]
GREEN = COLORS["mint"]
AMBER = COLORS["gold"]
BG = COLORS["surface"]
CARD = COLORS["panel"]

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Churn Intelligence Platform",
    assets_folder=os.path.join(BASE_DIR, "assets"),
    suppress_callback_exceptions=True,
)
server = app.server


CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.86)",
    font=dict(family="Aptos, Segoe UI, sans-serif", size=13, color=COLORS["text"]),
    margin=dict(l=24, r=24, t=50, b=24),
    hoverlabel=dict(
        bgcolor="#11253d",
        bordercolor="rgba(255,255,255,0.25)",
        font=dict(color="#f8fbff"),
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(0,0,0,0)",
    ),
)


def chart_card(title, description, graph_id):
    return dbc.Card(
        [
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.P(title, className="panel-eyebrow"),
                            html.H4(description, className="panel-title"),
                        ],
                        className="panel-heading",
                    ),
                    dcc.Graph(id=graph_id, config={"displayModeBar": False}, className="graph-frame"),
                ]
            )
        ],
        className="dash-panel h-100",
    )


KPI_ICONS = {
    "customers": _icon_data_uri(
        """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" fill="none"><circle cx="24" cy="24" r="8" stroke="#173556" stroke-width="4"/><circle cx="42" cy="26" r="6" stroke="#173556" stroke-width="4" opacity="0.8"/><path d="M12 48c2-7 8-11 16-11s14 4 16 11" stroke="#173556" stroke-width="4" stroke-linecap="round"/><path d="M36 48c1.5-5 5.5-8 11-8 3.6 0 6.7 1.3 9 4" stroke="#173556" stroke-width="4" stroke-linecap="round" opacity="0.8"/></svg>"""
    ),
    "churn": _icon_data_uri(
        """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" fill="none"><path d="M12 18h40" stroke="#173556" stroke-width="4" stroke-linecap="round"/><path d="M16 48l10-10 8 6 14-18" stroke="#173556" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/><path d="M40 26h8v8" stroke="#173556" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/></svg>"""
    ),
    "charges": _icon_data_uri(
        """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" fill="none"><rect x="12" y="16" width="40" height="32" rx="8" stroke="#173556" stroke-width="4"/><path d="M24 32h16" stroke="#173556" stroke-width="4" stroke-linecap="round"/><path d="M32 24v16" stroke="#173556" stroke-width="4" stroke-linecap="round"/><path d="M18 24h4M42 40h4" stroke="#173556" stroke-width="4" stroke-linecap="round" opacity="0.75"/></svg>"""
    ),
    "tenure": _icon_data_uri(
        """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" fill="none"><circle cx="32" cy="32" r="18" stroke="#173556" stroke-width="4"/><path d="M32 22v12l8 5" stroke="#173556" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/><path d="M24 10l-4 5M44 10l4 5" stroke="#173556" stroke-width="4" stroke-linecap="round"/></svg>"""
    ),
}


def page_shell(title, copy, children):
    return html.Div(
        [
            html.Div(
                [
                    dbc.Button("Back to overview", href="/", color="link", className="back-link px-0"),
                    html.H2(title, className="detail-title"),
                    html.P(copy, className="detail-copy"),
                ],
                className="detail-header",
            ),
            children,
            project_footer,
        ],
        className="detail-page",
    )


def kpi_card(title, value, accent_class, subtitle, icon):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    html.Img(src=KPI_ICONS[icon], alt=title, className="kpi-icon"),
                    className=f"kpi-glow {accent_class}",
                ),
                html.P(title, className="kpi-title"),
                html.Div(value, className="kpi-value"),
                html.P(subtitle, className="kpi-subtitle"),
            ]
        ),
        className="kpi-card h-100",
    )


def linked_kpi_card(title, value, accent_class, subtitle, href, icon):
    return dcc.Link(
        kpi_card(title, value, accent_class, subtitle, icon),
        href=href,
        className="kpi-link",
    )


hero = html.Div(
    [
        html.Div(
            [
                html.Div("Retention Intelligence", className="hero-tag"),
                html.H1("Customer churn intelligence platform", className="hero-title"),
                html.P(
                    "Explore churn drivers, compare retention risk, and run live customer-level predictions "
                    "from a polished business-facing analytics workspace.",
                    className="hero-copy",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("Model", className="mini-stat-label"),
                                html.Strong(META["model_name"], className="mini-stat-value"),
                            ],
                            className="mini-stat",
                        ),
                        html.Div(
                            [
                                html.Span("AUC", className="mini-stat-label"),
                                html.Strong(str(META["roc_auc"]), className="mini-stat-value"),
                            ],
                            className="mini-stat",
                        ),
                    ],
                    className="mini-stat-row",
                ),
            ],
            className="hero-copy-wrap",
        ),
        html.Div(
            [
                html.Div("What this dashboard helps with", className="insight-label"),
                html.Div(
                    [
                        dbc.Button(
                            [
                                html.Span("Spot high-risk contracts", className="insight-title"),
                                html.Small("Open the contract analysis page", className="insight-copy"),
                            ],
                            href="/contracts",
                            color="link",
                            className="insight-pill",
                        ),
                        dbc.Button(
                            [
                                html.Span("Compare service mix", className="insight-title"),
                                html.Small("Open the internet service page", className="insight-copy"),
                            ],
                            href="/services",
                            color="link",
                            className="insight-pill",
                        ),
                        dbc.Button(
                            [
                                html.Span("Estimate churn instantly", className="insight-title"),
                                html.Small("Open the live prediction page", className="insight-copy"),
                            ],
                            href="/predictor",
                            color="link",
                            className="insight-pill",
                        ),
                    ],
                    className="insight-pills",
                ),
            ],
            className="hero-aside",
        ),
    ],
    className="hero-shell",
)

project_spotlight = html.Div(
    [
        html.Div(
            [
                html.Div("Project Snapshot", className="spotlight-kicker"),
                html.H3("Built like a portfolio-ready analytics product", className="spotlight-title"),
                html.P(
                    "This project combines exploratory churn analysis, model performance visibility, and "
                    "interactive scenario testing in one end-to-end workflow.",
                    className="spotlight-copy",
                ),
            ],
            className="spotlight-lead",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Span("Dataset", className="spotlight-label"),
                        html.Strong(f"{len(df_raw):,} customer records", className="spotlight-value"),
                        html.P("Telecom churn data prepared for analytics and model-driven decision support.", className="spotlight-note"),
                    ],
                    className="spotlight-card",
                ),
                html.Div(
                    [
                        html.Span("Model Quality", className="spotlight-label"),
                        html.Strong(str(META["roc_auc"]), className="spotlight-value"),
                        html.P("ROC-AUC from the trained churn pipeline surfaced directly inside the UI.", className="spotlight-note"),
                    ],
                    className="spotlight-card",
                ),
                html.Div(
                    [
                        html.Span("Presentation Value", className="spotlight-label"),
                        html.Strong("Analytics + Prediction", className="spotlight-value"),
                        html.P("Designed to communicate business context, risk insight, and technical capability.", className="spotlight-note"),
                    ],
                    className="spotlight-card",
                ),
            ],
            className="spotlight-grid",
        ),
    ],
    className="project-spotlight mb-4",
)

project_footer = html.Div(
    [
        html.Div(
            [
                html.Div("Project Stack", className="footer-label"),
                html.Div("Dash, Plotly, Pandas, Scikit-learn", className="footer-value"),
            ],
            className="footer-item",
        ),
        html.Div(
            [
                html.Div("Use Case", className="footer-label"),
                html.Div("Telecom churn analytics and retention prediction", className="footer-value"),
            ],
            className="footer-item",
        ),
        html.Div(
            [
                html.Div("Presentation Ready", className="footer-label"),
                html.Div("Interactive, business-focused, and portfolio friendly", className="footer-value"),
            ],
            className="footer-item",
        ),
    ],
    className="project-footer",
)


predictor_panel = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.P("Action Studio", className="panel-eyebrow"),
                                    dbc.Button(
                                        "Live churn predictor",
                                        href="/predictor/live",
                                        color="link",
                                        className="predictor-title-btn",
                                    ),
                                    html.P(
                                        "Adjust customer attributes and evaluate churn probability in real time.",
                                        className="predictor-copy",
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    dbc.Button(
                                        "Professional scoring workspace",
                                        href="/predictor/workspace",
                                        id="workspace-chip",
                                        color="link",
                                        className="predictor-chip-btn",
                                    ),
                                    dbc.Button(
                                        "Retention decision support",
                                        href="/predictor/decision-support",
                                        id="decision-chip",
                                        color="link",
                                        className="predictor-chip-btn predictor-chip-btn-soft",
                                    ),
                                ],
                                className="predictor-chip-row",
                            ),
                        ],
                        className="predictor-topbar",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("01", className="studio-step-num"),
                                    html.Div(
                                        [
                                            html.Div("Profile the customer", className="studio-step-title"),
                                            html.Div("Set key contract, pricing, and tenure signals.", className="studio-step-copy"),
                                        ]
                                    ),
                                ],
                                className="studio-step",
                            ),
                            html.Div(
                                [
                                    html.Span("02", className="studio-step-num"),
                                    html.Div(
                                        [
                                            html.Div("Add support context", className="studio-step-title"),
                                            html.Div("Capture service behavior and support coverage.", className="studio-step-copy"),
                                        ]
                                    ),
                                ],
                                className="studio-step",
                            ),
                            html.Div(
                                [
                                    html.Span("03", className="studio-step-num"),
                                    html.Div(
                                        [
                                            html.Div("Review churn risk", className="studio-step-title"),
                                            html.Div("Generate an instant model-backed prediction.", className="studio-step-copy"),
                                        ]
                                    ),
                                ],
                                className="studio-step",
                            ),
                        ],
                        className="studio-steps",
                    ),
                ],
                className="predictor-header",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Core customer profile", className="form-section-title"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Label("Tenure (months)", className="control-label"),
                                                    dcc.Slider(
                                                        1,
                                                        72,
                                                        1,
                                                        value=12,
                                                        id="p-tenure",
                                                        marks={
                                                            1: "1",
                                                            12: "12",
                                                            24: "24",
                                                            36: "36",
                                                            48: "48",
                                                            60: "60",
                                                            72: "72",
                                                        },
                                                        tooltip={"placement": "bottom", "always_visible": True},
                                                        className="soft-slider",
                                                    ),
                                                ],
                                                md=6,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label("Monthly Charges ($)", className="control-label"),
                                                    dcc.Slider(
                                                        20,
                                                        120,
                                                        1,
                                                        value=70,
                                                        id="p-charges",
                                                        marks={
                                                            20: "20",
                                                            40: "40",
                                                            60: "60",
                                                            80: "80",
                                                            100: "100",
                                                            120: "120",
                                                        },
                                                        tooltip={"placement": "bottom", "always_visible": True},
                                                        className="soft-slider",
                                                    ),
                                                ],
                                                md=6,
                                            ),
                                        ],
                                        className="g-4 mb-3",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Label("Contract", className="control-label"),
                                                    dcc.Dropdown(
                                                        ["Month-to-month", "One year", "Two year"],
                                                        "Month-to-month",
                                                        id="p-contract",
                                                        clearable=False,
                                                        className="soft-dropdown",
                                                    ),
                                                ],
                                                md=4,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label("Internet Service", className="control-label"),
                                                    dcc.Dropdown(
                                                        ["DSL", "Fiber optic", "No"],
                                                        "Fiber optic",
                                                        id="p-internet",
                                                        clearable=False,
                                                        className="soft-dropdown",
                                                    ),
                                                ],
                                                md=4,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label("Payment Method", className="control-label"),
                                                    dcc.Dropdown(
                                                        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
                                                        "Electronic check",
                                                        id="p-payment",
                                                        clearable=False,
                                                        className="soft-dropdown",
                                                    ),
                                                ],
                                                md=4,
                                            ),
                                        ],
                                        className="g-4",
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Customer profile snapshot", className="profile-snapshot-title"),
                                            html.P(id="profile-snapshot-copy", className="profile-snapshot-copy"),
                                            html.Div(id="profile-snapshot-tags", className="profile-snapshot-tags"),
                                        ],
                                        className="profile-snapshot-panel mt-4",
                                    ),
                                ]
                            ),
                            className="studio-form-card",
                        ),
                        lg=8,
                    ),
                    dbc.Col(
                            dbc.Card(
                            dbc.CardBody(
                                [
                                    dbc.Button(
                                        "Prediction brief",
                                        href="/predictor/brief",
                                        color="link",
                                        className="form-section-title-btn",
                                    ),
                                    html.P(id="prediction-brief-copy", className="studio-brief-copy"),
                                    html.Div(id="prediction-brief-stats", className="brief-stat-grid"),
                                ]
                            ),
                            className="studio-side-card",
                        ),
                        lg=4,
                    ),
                ],
                className="g-4 mb-4 align-items-start",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Retention support signals", className="form-section-title"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Label("Tech Support", className="control-label"),
                                                    dcc.Dropdown(
                                                        ["Yes", "No", "No internet service"],
                                                        "No",
                                                        id="p-techsupport",
                                                        clearable=False,
                                                        className="soft-dropdown",
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label("Online Security", className="control-label"),
                                                    dcc.Dropdown(
                                                        ["Yes", "No", "No internet service"],
                                                        "No",
                                                        id="p-security",
                                                        clearable=False,
                                                        className="soft-dropdown",
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label("Senior Citizen", className="control-label"),
                                                    dcc.Dropdown(
                                                        ["Yes", "No"],
                                                        "No",
                                                        id="p-senior",
                                                        clearable=False,
                                                        className="soft-dropdown",
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label("Number of Services", className="control-label"),
                                                    dcc.Slider(
                                                        1,
                                                        7,
                                                        1,
                                                        value=2,
                                                        id="p-services",
                                                        marks={
                                                            1: "1",
                                                            2: "2",
                                                            3: "3",
                                                            4: "4",
                                                            5: "5",
                                                            6: "6",
                                                            7: "7",
                                                        },
                                                        tooltip={"placement": "bottom", "always_visible": True},
                                                        className="soft-slider",
                                                    ),
                                                ],
                                                md=3,
                                            ),
                                        ],
                                        className="g-4",
                                    ),
                                ]
                            ),
                            className="studio-form-card",
                        ),
                        lg=12,
                    ),
                ],
                className="g-4 mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Run prediction", className="form-section-title"),
                                    dbc.Button(
                                        "Predict churn",
                                        id="predict-btn",
                                        color="primary",
                                        size="lg",
                                        className="predict-btn w-100",
                                    ),
                                ]
                            ),
                            className="studio-cta-card h-100",
                        ),
                        md=4,
                    ),
                    dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.Div("Prediction output", className="form-section-title"),
                                        html.Div(
                                            [
                                                html.Div("Ready for scoring", className="prediction-title"),
                                                html.Div(
                                                    "Adjust the customer profile and click Generate churn forecast.",
                                                    className="prediction-line",
                                                ),
                                            ],
                                            id="predict-result",
                                            className="prediction-shell prediction-shell-empty",
                                        ),
                                    ]
                                ),
                                className="studio-result-card h-100",
                        ),
                        md=8,
                    ),
                ],
                className="g-4 align-items-stretch mt-2",
            ),
        ]
    ),
    className="predictor-panel mb-5",
)


overview_layout = html.Div(
    [
        html.Div(className="home-backdrop-grid"),
        html.Div(className="home-backdrop-glow glow-one"),
        html.Div(className="home-backdrop-glow glow-two"),
        html.Div(className="home-backdrop-glow glow-three"),
        hero,
        project_spotlight,
        html.Div(id="kpi-row", className="kpi-grid mb-4"),
        dbc.Row(
            [
                dbc.Col(
                    chart_card(
                        "Retention Signals",
                        "Churn by contract type",
                        "contract-chart",
                    ),
                    lg=6,
                    className="mb-4",
                ),
                dbc.Col(
                    chart_card(
                        "Revenue Pattern",
                        "Monthly charges distribution",
                        "charges-hist",
                    ),
                    lg=6,
                    className="mb-4",
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    chart_card(
                        "Connectivity Breakdown",
                        "Churn by internet service",
                        "internet-chart",
                    ),
                    lg=6,
                    className="mb-4",
                ),
                dbc.Col(
                    chart_card(
                        "Behavior Map",
                        "Tenure vs monthly charges",
                        "scatter-chart",
                    ),
                    lg=6,
                    className="mb-4",
                ),
            ]
        ),
        predictor_panel,
        project_footer,
    ],
    className="homepage-shell",
)


def build_contracts_page():
    return page_shell(
        "Contract Risk Analysis",
        "Use this view to understand which contract types are most exposed to churn and how pricing patterns shift between retained and churned customers.",
        dbc.Row(
            [
                dbc.Col(
                    chart_card(
                        "Retention Signals",
                        "Churn by contract type",
                        "contract-chart",
                    ),
                    lg=6,
                    className="mb-4",
                ),
                dbc.Col(
                    chart_card(
                        "Revenue Pattern",
                        "Monthly charges distribution",
                        "charges-hist",
                    ),
                    lg=6,
                    className="mb-4",
                ),
            ]
        ),
    )


def build_services_page():
    return page_shell(
        "Service Mix Analysis",
        "This page focuses on how internet service choices and customer behavior connect to churn risk across your telecom project data.",
        dbc.Row(
            [
                dbc.Col(
                    chart_card(
                        "Connectivity Breakdown",
                        "Churn by internet service",
                        "internet-chart",
                    ),
                    lg=6,
                    className="mb-4",
                ),
                dbc.Col(
                    chart_card(
                        "Behavior Map",
                        "Tenure vs monthly charges",
                        "scatter-chart",
                    ),
                    lg=6,
                    className="mb-4",
                ),
            ]
        ),
    )


def build_predictor_page():
    return page_shell(
        "Live Churn Prediction",
        "Simulate a customer profile and estimate whether that customer is likely to churn based on your trained model.",
        predictor_panel,
    )


def info_page(title, copy, bullets):
    return page_shell(
        title,
        copy,
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div("Overview", className="form-section-title"),
                    html.Div(
                        [html.Div(item, className="info-bullet") for item in bullets],
                        className="info-bullet-list",
                    ),
                    dbc.Button("Open Live Predictor", href="/predictor", color="primary", className="predict-btn mt-4"),
                ]
            ),
            className="dash-panel",
        ),
    )


def build_customers_page():
    return page_shell(
        "Customer Portfolio Overview",
        "Review the key customer base metrics from your churn project and use them as a starting point before drilling into risk, services, or prediction flows.",
        dbc.Row(
            [
                dbc.Col(html.Div(id="kpi-row-detail", className="kpi-grid-detail"), lg=12, className="mb-4"),
            ]
        ),
    )


app.layout = dbc.Container(
    [
        dcc.Location(id="url"),
        html.Div(className="bg-orb orb-a"),
        html.Div(className="bg-orb orb-b"),
        html.Div(id="page-content"),
    ],
    fluid=True,
    className="app-shell py-4 py-lg-5",
)


@app.callback(Output("kpi-row", "children"), Input("kpi-row", "id"))
def kpi(_):
    total = len(df_raw)
    churned = int(df_raw["churn"].sum())
    churn_rate = churned / total
    avg_charge = df_raw.loc[df_raw["churn"] == 1, "monthly_charges"].mean()
    avg_tenure = df_raw.loc[df_raw["churn"] == 0, "tenure"].mean()

    cards = [
        (
            linked_kpi_card(
                "Total Customers",
                f"{total:,}",
                "accent-blue",
                "Open the customer overview page",
                "/customers",
                "customers",
            ),
            {"xl": 3, "md": 6},
        ),
        (
            linked_kpi_card(
                "Churned Customers",
                f"{churned:,}",
                "accent-red",
                f"Open contract risk page • {churn_rate:.1%} churn rate",
                "/contracts",
                "churn",
            ),
            {"xl": 3, "md": 6},
        ),
        (
            linked_kpi_card(
                "Avg Charges of Churned",
                f"${avg_charge:.2f}",
                "accent-gold",
                "Open service mix page",
                "/services",
                "charges",
            ),
            {"xl": 3, "md": 6},
        ),
        (
            linked_kpi_card(
                "Avg Tenure of Retained",
                f"{avg_tenure:.1f} mo",
                "accent-green",
                "Open live predictor page",
                "/predictor",
                "tenure",
            ),
            {"xl": 3, "md": 6},
        ),
    ]
    return [dbc.Col(card, className="mb-3", **sizes) for card, sizes in cards]


@app.callback(Output("kpi-row-detail", "children"), Input("kpi-row-detail", "id"))
def kpi_detail(_):
    return kpi("kpi-row-detail")


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname):
    if pathname == "/customers":
        return build_customers_page()
    if pathname == "/contracts":
        return build_contracts_page()
    if pathname == "/services":
        return build_services_page()
    if pathname == "/predictor/live":
        return info_page(
            "Live Churn Predictor",
            "This page explains how the live predictor works in your churn project and what the prediction output is meant to help you decide.",
            [
                "Use the predictor to test individual customer profiles before taking a retention action.",
                "The model estimates churn probability from tenure, billing, contract, service, and support inputs.",
                "This is useful for scenario analysis such as checking whether adding support or changing contract type reduces expected churn risk.",
            ],
        )
    if pathname == "/predictor/brief":
        return info_page(
            "Prediction Brief",
            "The prediction brief summarizes the current customer scenario in plain language before you run the model.",
            [
                "It updates live from the current form inputs.",
                "It highlights billing level, contract stability, support posture, customer segment, and payment context.",
                "This gives you a quick business summary before generating the full churn forecast.",
            ],
        )
    if pathname == "/predictor/workspace":
        return info_page(
            "Professional Scoring Workspace",
            "This area is designed as the main working space for building customer scenarios and preparing a professional prediction workflow.",
            [
                "It organizes the inputs into profile, account configuration, and support signal sections.",
                "The goal is to make the churn prediction process easier to read and more decision-oriented.",
                "Use it when you want a cleaner workflow for testing multiple retention scenarios.",
            ],
        )
    if pathname == "/predictor/decision-support":
        return info_page(
            "Retention Decision Support",
            "This section connects the model output to business decisions so the forecast is useful beyond just a raw percentage.",
            [
                "It helps identify whether a customer looks stable, medium-risk, or high-risk.",
                "You can use the result to decide when to prioritize outreach, offers, contract changes, or support improvements.",
                "It is meant to support retention planning, not just display a model score.",
            ],
        )
    if pathname == "/predictor":
        return build_predictor_page()
    return overview_layout


@app.callback(Output("contract-chart", "figure"), Input("contract-chart", "id"))
def contract_chart(_):
    grp = df_raw.groupby(["contract", "churn"]).size().reset_index(name="count")
    grp["churn_label"] = grp["churn"].map({0: "No Churn", 1: "Churn"})
    fig = px.bar(
        grp,
        x="contract",
        y="count",
        color="churn_label",
        barmode="group",
        color_discrete_map={"Churn": RED, "No Churn": BLUE},
        text_auto=True,
    )
    fig.update_traces(marker_line_width=0, hovertemplate="%{x}<br>%{fullData.name}: %{y}<extra></extra>")
    fig.update_layout(
        **CHART_LAYOUT,
        legend_title="",
        xaxis_title="Contract Type",
        yaxis_title="Customers",
        yaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
        xaxis=dict(showgrid=False),
    )
    return fig


@app.callback(Output("charges-hist", "figure"), Input("charges-hist", "id"))
def charges_hist(_):
    fig = go.Figure()
    for label, color in [(0, BLUE), (1, RED)]:
        subset = df_raw[df_raw["churn"] == label]["monthly_charges"]
        fig.add_trace(
            go.Histogram(
                x=subset,
                name="No Churn" if label == 0 else "Churn",
                opacity=0.72,
                marker_color=color,
                nbinsx=30,
                hovertemplate="Monthly Charges: %{x}<br>Customers: %{y}<extra></extra>",
            )
        )
    fig.update_layout(
        **CHART_LAYOUT,
        barmode="overlay",
        legend_title="",
        xaxis_title="Monthly Charges ($)",
        yaxis_title="Customers",
        yaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
        xaxis=dict(showgrid=False),
    )
    return fig


@app.callback(Output("internet-chart", "figure"), Input("internet-chart", "id"))
def internet_chart(_):
    grp = (
        df_raw[df_raw["churn"] == 1]
        .groupby("internet_service")
        .size()
        .reset_index(name="churned")
    )
    fig = px.pie(
        grp,
        names="internet_service",
        values="churned",
        hole=0.48,
        color="internet_service",
        color_discrete_sequence=[BLUE, RED, AMBER],
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker=dict(line=dict(color="#ffffff", width=2)),
        hovertemplate="%{label}<br>Churned Customers: %{value}<br>Share: %{percent}<extra></extra>",
    )
    fig.update_layout(**CHART_LAYOUT, legend_title="")
    return fig


@app.callback(Output("scatter-chart", "figure"), Input("scatter-chart", "id"))
def scatter(_):
    sample = df_raw.sample(min(1500, len(df_raw)), random_state=42).copy()
    sample["churn_label"] = sample["churn"].map({0: "No Churn", 1: "Churn"})
    fig = px.scatter(
        sample,
        x="tenure",
        y="monthly_charges",
        color="churn_label",
        color_discrete_map={"Churn": RED, "No Churn": BLUE},
        opacity=0.7,
        size_max=7,
    )
    fig.update_traces(
        marker=dict(size=9, line=dict(width=1, color="rgba(255,255,255,0.6)")),
        hovertemplate="Tenure: %{x} mo<br>Monthly Charges: $%{y:.2f}<br>%{fullData.name}<extra></extra>",
    )
    fig.update_layout(
        **CHART_LAYOUT,
        legend_title="",
        xaxis_title="Tenure (months)",
        yaxis_title="Monthly Charges ($)",
        yaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
        xaxis=dict(gridcolor="rgba(219,228,240,0.45)", zeroline=False),
    )
    return fig


@app.callback(
    Output("workspace-chip", "children"),
    Output("decision-chip", "children"),
    Output("prediction-brief-copy", "children"),
    Output("prediction-brief-stats", "children"),
    Output("profile-snapshot-copy", "children"),
    Output("profile-snapshot-tags", "children"),
    Input("p-tenure", "value"),
    Input("p-charges", "value"),
    Input("p-contract", "value"),
    Input("p-internet", "value"),
    Input("p-payment", "value"),
    Input("p-techsupport", "value"),
    Input("p-security", "value"),
    Input("p-senior", "value"),
    Input("p-services", "value"),
)
def update_prediction_brief(tenure, charges, contract, internet, payment, techsupport, security, senior, services):
    service_level = "high-service" if services >= 5 else ("balanced-service" if services >= 3 else "lean-service")
    price_band = "premium billing" if charges >= 85 else ("mid billing" if charges >= 55 else "value billing")
    contract_signal = "stable contract" if contract != "Month-to-month" else "flex contract"

    risk_flags = []
    if contract == "Month-to-month":
        risk_flags.append("month-to-month exposure")
    if internet == "Fiber optic":
        risk_flags.append("fiber segment")
    if charges >= 85:
        risk_flags.append("high monthly spend")
    if tenure <= 12:
        risk_flags.append("newer customer")
    if techsupport == "No":
        risk_flags.append("no tech support")
    if security == "No":
        risk_flags.append("no online security")

    if risk_flags:
        brief_copy = (
            "This scenario highlights "
            + ", ".join(risk_flags[:3])
            + ". Use the prediction output to check whether this profile needs proactive retention outreach."
        )
    else:
        brief_copy = (
            "This customer profile looks comparatively stable based on the selected contract, pricing, "
            "and support settings. Generate a forecast to confirm retention risk."
        )

    senior_text = "Senior segment" if senior == "Yes" else "General segment"
    support_text = "Protected account" if techsupport == "Yes" and security == "Yes" else "Support gap present"

    brief_stats = [
        html.Div(
            [
                html.Span("Workspace", className="brief-kicker"),
                html.Strong(f"{service_level.replace('-', ' ').title()}"),
            ],
            className="brief-stat",
        ),
        html.Div(
            [
                html.Span("Billing", className="brief-kicker"),
                html.Strong(price_band.title()),
            ],
            className="brief-stat",
        ),
        html.Div(
            [
                html.Span("Contract", className="brief-kicker"),
                html.Strong(contract_signal.title()),
            ],
            className="brief-stat",
        ),
        html.Div(
            [
                html.Span("Support", className="brief-kicker"),
                html.Strong(support_text),
            ],
            className="brief-stat",
        ),
        html.Div(
            [
                html.Span("Segment", className="brief-kicker"),
                html.Strong(senior_text),
            ],
            className="brief-stat",
        ),
        html.Div(
            [
                html.Span("Payment", className="brief-kicker"),
                html.Strong(payment),
            ],
            className="brief-stat",
        ),
    ]

    workspace_chip = f"{contract_signal.title()} | {price_band.title()}"
    decision_chip = f"Decision Support | {internet} | {services} services"
    lifecycle = "new customer" if tenure <= 12 else ("growing customer" if tenure <= 36 else "established customer")
    payment_short = payment.replace(" transfer", "").replace("Electronic ", "E-").replace("Credit ", "Card ")
    snapshot_copy = (
        f"This profile represents a {lifecycle} on a {contract.lower()} plan with {internet.lower()} service, "
        f"{services} active services, and {price_band}. The current setup is useful for testing retention risk "
        f"before outreach or offer design."
    )
    snapshot_tags = [
        html.Span(f"{tenure} mo tenure", className="profile-tag"),
        html.Span(contract, className="profile-tag"),
        html.Span(internet, className="profile-tag"),
        html.Span(f"${charges} monthly", className="profile-tag"),
        html.Span(payment_short, className="profile-tag"),
        html.Span(support_text, className="profile-tag"),
    ]

    return workspace_chip, decision_chip, brief_copy, brief_stats, snapshot_copy, snapshot_tags


@app.callback(
    Output("predict-result", "children"),
    Input("predict-btn", "n_clicks"),
    State("p-tenure", "value"),
    State("p-charges", "value"),
    State("p-contract", "value"),
    State("p-internet", "value"),
    State("p-payment", "value"),
    State("p-techsupport", "value"),
    State("p-security", "value"),
    State("p-senior", "value"),
    State("p-services", "value"),
    prevent_initial_call=True,
)
def predict(_, tenure, charges, contract, internet, payment, techsupport, security, senior, services):
    if not MODEL_LOADED:
        return dbc.Alert("Model not loaded. Run `python ml/train.py` first.", color="warning")

    raw = {
        "tenure": tenure,
        "monthly_charges": charges,
        "total_charges": round(charges * tenure, 2),
        "num_services": services,
        "senior_citizen": 1 if senior == "Yes" else 0,
        "contract": contract,
        "internet_service": internet,
        "payment_method": payment,
        "tech_support": techsupport,
        "online_security": security,
        "paperless_billing": "Yes",
        "dependents": "No",
        "partner": "No",
    }

    row = pd.DataFrame([raw])
    row["charge_per_service"] = (row["monthly_charges"] / row["num_services"]).round(2)
    row["high_value"] = (row["monthly_charges"] > 80).astype(int)
    row["tenure_group"] = pd.cut(
        row["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"],
    )

    for col, encoder in ENCODERS.items():
        if col in row.columns:
            try:
                row[col] = encoder.transform(row[col].astype(str))
            except Exception:
                row[col] = 0

    row = row.reindex(columns=FEATURE_NAMES, fill_value=0)
    X = SCALER.transform(row)

    prob = float(MODEL.predict_proba(X)[0][1])
    label = int(prob >= 0.5)
    risk = "High Risk" if prob >= 0.70 else ("Medium Risk" if prob >= 0.40 else "Low Risk")
    outcome = "Will Churn" if label else "Will Retain"
    tone = "danger" if label else "success"

    return dbc.Alert(
        [
            html.Div(outcome, className="prediction-title"),
            html.Div(f"Churn Probability: {prob:.1%}", className="prediction-line"),
            html.Div(f"Risk Level: {risk}", className="prediction-line"),
        ],
        color=tone,
        className="prediction-alert mb-0",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"Dashboard running -> http://localhost:{port}")
    app.run(debug=True, host="0.0.0.0", port=port, use_reloader=False)
