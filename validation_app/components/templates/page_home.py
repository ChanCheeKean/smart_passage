from dash import html, dcc
import dash_bootstrap_components as dbc
# from components.callbacks import page_1_cb

# jumbotron
jumbotron = html.Div(
    dbc.Container(
        [
            html.H1("Performance Validation Framework", className="display-3"),
            html.P(
                "A tool to collect, label, evaluate and visualize"
                "the performance of the model or solution",
                className="lead",
            ),
            html.Hr(className="my-2"),
            html.P(
                "..."
            ),
            html.P(
                dbc.Button("Learn more", color="primary"), className="lead"
            ),
        ],
        fluid=True,
        className="py-3",
    ),
    className="p-3 bg-light rounded-3 mt-5",
)

### combination of all output
layout = html.Div(
        children=[
            jumbotron,
        ],
    )