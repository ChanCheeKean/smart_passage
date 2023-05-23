from dash import Input, Output, State
from app import app
from components.templates import page_error, page_home, page_collect, page_annotate

### Collapsible side bar, trigger by clickable hamburger icon
@app.callback(
    Output("offcanvas", "is_open"),
    Input("side_ham_button", "n_clicks"),
    State("offcanvas", "is_open"),
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

### update page content, trigger by sidebar selection
@app.callback(
    Output("main_content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if (pathname == "/home/") | (pathname == "/") :
        return page_home.layout
    elif pathname == "/collect/":
        return page_collect.layout
    elif pathname == "/annotate/":
        return page_annotate.layout
    else:
        return page_error.layout