from dash import html
import dash_bootstrap_components as dbc

layout = dbc.Container([
    html.H1('Bem-vindo ao Dashboard de Análise de Risco de Fogo - Cerrado', className='text-center my-4'),
    html.P('Nesta página, explicaremos a estrutura de dados utilizada para analisar casos de incêndio e seu risco de fogo.' +
           ' Use a barra de navegação acima para acessar o Mapa Interativo e os Grafos por Ano.', className='text-center')
], className='mt-5')
