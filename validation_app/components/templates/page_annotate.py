from dash import html, dcc
import dash_bootstrap_components as dbc
from components.style import general
from components.callbacks import annotate_cb

### image carousell
img_carousel = dbc.Carousel(
    id='img_carousel',
    items=[],
    controls=True,
    indicators=True,
    # variant="dark",
)

### image plotter
# config = {
#     "modeBarButtonsToAdd": [
#     "drawline",
#     "drawopenpath",
#     "drawclosedpath",
#     "drawcircle",
#     "drawrect",
#     "eraseshape",
#     ]
# }

# img_display = html.Div(
#     children=dcc.Graph(
#         id='annotate_img', 
#         config=config,
#         ),
#     className='w-100 h-100',
#     style={'padding' : '0', 'height' : '82vh',}
# )

### drop down for batch library
library_dropdown = dbc.Select(
    id="library_dropdown",
    style=general.dropdown_selector,
    className='d-inline mx-1',
)

library_select = html.Div(
    children=[
        html.P("Import Library: ", className="fs-6 fw-bold d-inline"), 
        library_dropdown
    ],
    className='d-inline',
)

img_card = dbc.Card(
    children=[
        dbc.CardHeader(library_select),
        dbc.CardBody(
            children=[img_carousel], 
            style={'padding' : '0', 'height' : '88vh',}
        ),
    ],
)

### metadata ###

# unlabel toggle
repeat_toggle = dbc.Switch(
    id="label_switch",
    label="Only Show Unlabeled",
    value=True,
    className="fs-6 fw-bold"
    )

# save button
save_but = dbc.Button(
    children="Save ➠", 
    id='save_annotate_but',
    outline=False, 
    disabled=False,
    color="primary",
    size="md", 
    className="w-100",
    n_clicks=0
)

# delete button
delete_but = dbc.Button(
    children="Delete ✂", 
    id='delete_annotate_but',
    outline=False, 
    disabled=False,
    color="danger",
    size="md", 
    className="w-100",
    n_clicks=0
)

# append button
append_but = dbc.Button(
    children="Append ✎", 
    id='append_annotate_but',
    outline=False, 
    disabled=False,
    color="success",
    size="md", 
    className="w-100",
    n_clicks=0
)

input_but = dbc.Row(
    children=[
        dbc.Col([save_but], width=6, className='px-1'),
        dbc.Col([append_but], width=3, className='px-1'),
        dbc.Col([delete_but], width=3, className='px-1'),
        ],
    className='mt-2')

# label input
class_dropdown = dbc.Select(
    id="class_dropdown",
    style=general.dropdown_selector,
    options=[{'label': key.title(),'value' : val} for key, val 
             in zip(['human', 'object', 'None'], [0, 1, -1]) ],
    value='0',
    className='d-inline mx-1',
)

class_select = html.Div(
    children=[
        html.P("Class: ", className="fs-6 fw-bold d-inline"), 
        class_dropdown
    ],
    className='d-inline',
)

# zone label
zone_dropdown = dbc.Select(
    id="zone_dropdown",
    style=general.dropdown_selector,
    options=[{'label': key.title(),'value' : val} for key, val 
             in zip(['left', 'safety', 'right', 'None'], [0, 1, 2, -1]) ],
    value='0',
    className='d-inline mx-1',
)

zone_select = html.Div(
    children=[
        html.P("Zone: ", className="fs-6 fw-bold d-inline"), 
        zone_dropdown
    ],
    className='my-3',
)

# desc
desc_input = html.Div(
    children=[
        dbc.Label("Desc: ", className="fs-6 fw-bold d-inline"),
        dbc.Input(id='textinput_desc', value='', type="text", className='d-inline w-75',),
    ],
    className='my-3',
)

# saved output
saved_input = html.Div(
    children=[
        dbc.Label("Saved Output: ", className="fs-6 fw-bold"),
        dbc.Textarea(id='textinput_savedjson', value='', className='', size="lg", disabled=True),
    ],
    className='my-3',
)

info_card = dbc.Card(
    children=[
        dbc.CardHeader(html.P(id='img_annotate_header', className="fs-6 fw-bold my-2")),
        dbc.CardBody(
            children=[
                class_select,
                zone_select,
                desc_input,
                saved_input,
                repeat_toggle,
                input_but,
                ], 
            style={'height' : '88vh'},
        ),
    ],
)

### final layout
layout = html.Div(
        children=[
            dbc.Row(
                children=[
                    dbc.Col([img_card], width=9),
                    dbc.Col([info_card], width=3),
                    ],
                className='my-1'),
            html.Div(id="annotate_test"),
        ]
    )