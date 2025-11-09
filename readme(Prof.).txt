Para execução do Dash -

1 - Após realizar git clone no projeto, no diretório da página, criar uma pasta .venv

    Windows:
        py -m venv .venv
    
    Linux / macOS:
        python3 -m venv .venv


2 - Após gerar a pasta .venv, será necessário acessar o ambiente virtual

    Windows:
        .venv\Scripts\activate
    
    Linux / macOS:
        source .venv/bin/activate
        # ou
        . .venv/bin/activate


3 - Após aparecer (.venv) por trás da path do terminal, instale os requerimentos do programa

    Windows:
        pip install -r requirements.txt

    Linux / macOS:
        pip install -r requirements.txt
        # se necessário:
        pip3 install -r requirements.txt


4 - Concluído a instalação, pode rodar o arquivo principal que vai rodar a renderização do Dash

    Windows:
        py main.py

    Linux / macOS:
        python3 main.py
        # ou
        python main.py