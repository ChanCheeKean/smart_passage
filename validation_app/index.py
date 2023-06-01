from app import app
from components.templates.main import make_layout
app.layout = make_layout()

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=False)
