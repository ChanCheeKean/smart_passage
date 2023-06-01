from dash import html, dcc
import dash_bootstrap_components as dbc
from components.style import general
from components.callbacks import validate_cb

### library dropdown
library_dropdown = dbc.Select(
    id="validate_library_dropdown",
    style=general.dropdown_selector,
    className='d-inline mx-1',
)

### button
generate_but = dbc.Button(
    children="Validate", 
    id='validate_but',
    outline=False, 
    disabled=False,
    color="primary", 
    size="sm", 
    className="d-inline mx-2 pb-1",
    n_clicks=0
)

input_select = html.Div(
    children=[
        html.P("Validate Library: ", className="fs-6 fw-bold d-inline"), 
        library_dropdown,
        generate_but,
    ],
    className='d-inline my-3',
)

### bar chart
bar_chart = dcc.Loading(
    children=dcc.Graph(
        id='validate_bar', 
        style=dict(height='60vh'),
    ),
    type="default")

### bar chart
pie_chart = dcc.Loading(
    children=dcc.Graph(
        id='validate_pie', 
        style=dict(height='60vh'),
    ),
    type="default")

### final layout
layout = html.Div(
        children = [
            input_select,
            html.Div(id='validate_store', style={'display': 'none'}),
            dbc.Row(
                children=[
                    dbc.Col([bar_chart], width=6, className='px-1'),
                    dbc.Col([pie_chart], width=6, className='px-1'),
                    ],
                className='my-5'),
        ]
    )