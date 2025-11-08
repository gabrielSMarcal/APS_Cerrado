from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# Inicialização da aplicação Dash
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        'assets/main.css'
    ],
    suppress_callback_exceptions=True
)

# Importar pages depois de criar o app para evitar importação circular
import pages

# --- Layout da Barra de Navegação ---
navegacao = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink('Início', href='/')),
        dbc.NavItem(dbc.NavLink('Gráficos por Ano', href='/graficos')),
        dbc.NavItem(dbc.NavLink('Modelo de previsão', href='/dados')),
        dbc.NavItem(dbc.NavLink('Previsão de 2026', href='/mapa')),
    ],
    brand='Análise de Risco de Fogo - Cerrado',
    brand_href='/',
    color='primary',
    dark=True,
    className='mb-5'
)

# --- Layout Principal ---
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navegacao,
    html.Div(id='conteudo', className='container-fluid')
])

# --- Callback de Roteamento ---
@app.callback(
    Output('conteudo', 'children'),
    Input('url', 'pathname')
)
def render_page_content(pathname):
    if pathname == '/':
        return pages.home.layout
    elif pathname == '/mapa':
        return pages.mapa.layout
    elif pathname == '/graficos':
        return pages.graficos.layout
    elif pathname == '/dados':
        return pages.dados.layout
    
    # Página não encontrada (404)
    return html.Div(
        [
            html.H1('404: Not found', className='text-danger'),
            html.Hr(),
            html.P(f'O caminho {pathname} não foi reconhecido...'),
        ],
        className='p-3 bg-light rounded-3',
    )
