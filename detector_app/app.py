import dash
import dash_bootstrap_components as dbc

stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(
        __name__,
        external_stylesheets=stylesheets,
        suppress_callback_exceptions=True,
)
server = app.server