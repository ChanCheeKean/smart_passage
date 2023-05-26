from dash import html
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

### final layout
layout = html.Div(
        children = [
            input_select,
            html.Div(id='validate_dummy'),
            # dbc.Row(
            #     children=[
            #         dbc.Col([video_card], width=9),
            #         dbc.Col([archive_card], width=3),
            #         ],
            #     className='my-1'),
        ]
    )