from typing import Dict, List, Any

class Graph:
    '''
    Classe abstrada do TAD Gráfico (Graph)
    '''
    
    def __init__(self):
        self._graph: Dict[Any, Dict[Any, float]] = {}

    def add_ponto(self, ponto: Any) -> None:
        '''
        Adiciona um ponto ao gráfico se ele não existe
        '''
        
        if ponto not in self._graph:
            self._graph[ponto] = {}

    def add_ponto_con(self, u: Any, v: Any, peso: float = 1.0) -> None:
        '''
        Faz a conexão de dois pontos com um peso
        Adicona dois pontos caso não existam
        '''
        
        self.add_ponto(u)
        self.add_ponto(v)
        
        self._graph[u][v] = peso
        self._graph[v][u] = peso

    def get_vizinhos(self, ponto: Any) -> List[Any]:
        '''
        Retorna os vizinhos de um ponto
        '''
        
        if ponto in self._graph:
            return list(self._graph[ponto].keys())
        return []

    def get_peso(self, u: Any, v: Any) -> float:
        '''
        Retorna o peso da conexão entre dois pontos
        '''
        
        if u in self._graph and v in self._graph[u]:
            return self._graph[u][v]
        return 0.0

    def get_pontos(self) -> List[Any]:
        '''
        Retorna todos os pontos do gráfico
        '''
        return list(self._graph.keys())

    def __len__(self) -> int:
        '''
        Retorna a quantidade de pontos em um gráfico
        '''
        return len(self._graph)

    def __contains__(self, ponto: Any) -> bool:
        '''
        Verifica se o ponto está no gráfico
        '''
        return ponto in self._graph

    def __repr__(self) -> str:
        '''
        Apresentão em string do gráfico
        '''
        return f"Pontos do gráfico: {len(self)} | Conexões de pontos: {self.total_ponto_con()}"

    def total_ponto_con(self) -> int:
        '''
        Apresenta a quantidade de conexões únicas no gráfico
        '''
        
        count = 0
        for u in self._graph:
            # Undirected graph, so we only count edges once
            for v in self._graph[u]:
                if u < v: # Assumes comparable keys for unique counting
                    count += 1
        return count