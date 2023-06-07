from app import app
# from components.pg1 import layout
# app.layout = layout

from dash import html, dcc
app.layout = html.Div()

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=5000,debug=False)