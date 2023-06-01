from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import date, datetime
from app import app
from components.style import general
from components.callbacks import main_cb

### top bar ###
top_right_banner = html.Div(
    children=[
        html.Span('Smart Passage Validation', className='text-black fw-bolder fs-6', id='app_name'),
        html.Sub('v0.01', className='fst-italic'),
        html.Span(' | ', className='text-black lead'),
        html.Span(id="current_date", className='text-black-50'),
        html.Span('  ', className='text-white'),
        html.Span(id = "current_time", className='text-black-50')], 
    className='float-end p-2'
)

top_bar = html.Div(
    children=[
        top_right_banner,
        dbc.Button(
            html.Img(src=app.get_asset_url('thales-logo.png'), style=general.logo),
            href='/',
            className='reset pointer',
        )
            ],
    style=general.top_bar
)

### side bar ###
side_bar = html.Div(
    children=[
        html.Button(
            html.Img(src=app.get_asset_url('hamburger-menu.png'), style=general.ham_icon),
            id="side_ham_button", 
            style=general.ham_button,
            className='hover-dim',
            ),

        dbc.Offcanvas(
            children=[
                dbc.Nav(
                    children=[
                        dbc.NavItem(dbc.NavLink(k, href=v, style=general.menu_font, className='hover-red')
                        ) for k, v in {
                                '⟰ Home' : '/home/',
                                '▣ Data Collection' : '/collect/',
                                '✒ Data Annotation' : '/annotate/',
                                '⌨ Data Validation' : '/validate/',
                                }.items()
                            ],
                    vertical="sm",
                    )
            ],
            placement='end',
            id="offcanvas",
            title="",
            is_open=False,
            className='fw-bolder fs-5',
            style=general.side_bar,
        ),
    ]
)

### content page
content = html.Div(id='main_content', className='px-4 py-5')

### Final Combination ###
def make_layout():
    layout = html.Div(
        children = [
            dcc.Location(id="url"),
            html.Div(id="dummy_div"),
            dcc.Interval(id='clock_interval_30', interval=30*1000, n_intervals=0),
            top_bar,
            side_bar, 
            content,
            ]
    )

    return layout

