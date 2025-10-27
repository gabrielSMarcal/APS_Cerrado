from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from app import app 
import pages
import os

# --- Layout da Barra de Navegação (Semelhante ao MachineLearningDashboard) ---
navegacao = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Mapa Interativo", href="/mapa")),
        dbc.NavItem(dbc.NavLink("Grafos por Ano", href="/grafos")),
        dbc.NavItem(dbc.NavLink("Início", href="/")),
    ],
    brand="Análise de Risco de Fogo - Cerrado",
    brand_href="/",
    color="primary",  # Cor primária do Bootstrap
    dark=True,
    className="mb-5"
)

# --- Layout Principal (Onde o conteúdo dinâmico será carregado) ---
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navegacao,
    html.Div(id='conteudo', className="container-fluid") # ID 'conteudo' para o conteúdo da página
])

# --- Callback de Roteamento ---
@app.callback(
    Output('conteudo', 'children'),
    [Input('url', 'pathname')]
)
def render_page_content(pathname):
    if pathname == "/":
        # Retorna o layout da página inicial
        return pages.home.layout
    elif pathname == "/mapa":
        # Retorna o layout da página do mapa
        return pages.mapa.layout
    elif pathname == "/grafos":
        # Retorna o layout da página de grafos
        return pages.grafos.layout
    
    # Página não encontrada (404)
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"O caminho {pathname} não foi reconhecido..."),
        ],
        className="p-3 bg-light rounded-3",
    )

if __name__ == '__main__':
    # Novo ponto de execução
    app.run(debug=True)
