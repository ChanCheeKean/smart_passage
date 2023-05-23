from dash import html, dcc
import dash_bootstrap_components as dbc
# from components.callbacks import page_1_cb

# jumbotron
jumbotron = html.Div(
    dbc.Container(
        [
            html.H1("Error", className="display-3"),
            html.P(
                "Unable to locate route",
                className="lead",
            ),
            html.Hr(className="my-2"),
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