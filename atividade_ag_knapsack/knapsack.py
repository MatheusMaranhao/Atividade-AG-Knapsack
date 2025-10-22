import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy
import time # Para marcar o tempo de execução

### ----------------------------------------------------
### PARTE 1: DEFINIÇÃO DO PROBLEMA (KNAPSACK DA ATIVIDADE 1)
### ----------------------------------------------------

def _knapsack_constants(dim):
    """Retorna os ganhos, pesos e capacidade da mochila."""
    GANHOS = []
    PESOS = []
    CAPACIDADE_MAXIMA = 0

    if dim == 20:
        # Estes são os dados da sua Atividade 1
        GANHOS = [92, 4, 43, 83, 84, 68, 92, 82, 6, 44, 32, 18, 56, 83, 25, 96, 70, 48, 14, 58]
        PESOS = [44, 46, 90, 72, 91, 40, 75, 35, 8, 54, 78, 40, 77, 15, 61, 17, 75, 29, 75, 63]
        CAPACIDADE_MAXIMA = 878
    else:
        # Só para garantir, caso eu mude o N_DIMENSOES
        raise ValueError("Dimensão não suportada. Use dim=20.")

    return GANHOS, PESOS, CAPACIDADE_MAXIMA

def knapsack(solution, dim=20):
    """
    Avalia uma solução para o problema da mochila.
    O 'ganho_total' será 0 se o peso estourar a capacidade.
    """
    GANHOS, PESOS, CAPACIDADE_MAXIMA = _knapsack_constants(dim)

    ganho_total = 0
    peso_total = 0

    for i in range(len(solution)):
        if solution[i] == 1:
            ganho_total += GANHOS[i]
            peso_total += PESOS[i]

    # A regra de penalidade da Atividade 1
    if peso_total > CAPACIDADE_MAXIMA:
        ganho_total = 0

    return ganho_total, peso_total

# -----------------------------------------------------------------
# FUNÇÃO DE FITNESS (USADA POR TODOS OS ALGORITMOS)
# -----------------------------------------------------------------

def calcular_fitness(solucao):
    """
    Função de fitness unificada. 
    Retorna apenas o ganho (valor) da mochila.
    """
    ganho, _ = knapsack(solucao, dim=N_DIMENSOES)
    return ganho

### ----------------------------------------------------
### PARTE 2: HILL CLIMBING (CÓDIGO DA ATIVIDADE 3)
### ----------------------------------------------------

def gerar_vizinhos_hc(solucao, n_vizinhos=20):
    """
    Gera vizinhos trocando um bit (bit-flip) aleatoriamente.
    """
    vizinhos = []
    n_itens = len(solucao)
    posicoes_usadas = [] # Evita sortear a mesma posição

    max_vizinhos = min(n_vizinhos, n_itens)

    while len(vizinhos) < max_vizinhos:
        pos = random.randint(0, n_itens - 1)
        if pos not in posicoes_usadas:
            posicoes_usadas.append(pos)
            vizinho = solucao.copy()
            vizinho[pos] = 1 - vizinho[pos] # Flip (0 vira 1, 1 vira 0)
            vizinhos.append(vizinho)
        
        # Se já usei todas as posições, não tem mais o que fazer
        if len(posicoes_usadas) == n_itens:
            break
            
    return vizinhos

class HillClimbing:
    """ Classe do Hill Climbing (versão flexível da Atividade 3). """
    def __init__(self, stochastic=False):
        self.stochastic = stochastic # Define se é Estocástico ou Tradicional
        self.historico = []

    def executar(self, solucao_inicial, max_iteracoes):
        solucao_atual = copy.deepcopy(solucao_inicial)
        fitness_atual = calcular_fitness(solucao_atual)
        self.historico = [fitness_atual]

        for _ in range(max_iteracoes):
            vizinhos = gerar_vizinhos_hc(solucao_atual)
            melhor_vizinho = None

            if self.stochastic:
                # MODO ESTOCÁSTICO: Sorteia um dos vizinhos que são melhores
                vizinhos_melhores = []
                for v in vizinhos:
                    fitness_v = calcular_fitness(v)
                    if fitness_v > fitness_atual:
                        vizinhos_melhores.append(v)
                
                if vizinhos_melhores:
                    melhor_vizinho = random.choice(vizinhos_melhores)
            else:
                # MODO TRADICIONAL: Pega o *melhor* de todos os vizinhos
                melhor_fitness_vizinho = fitness_atual
                for v in vizinhos:
                    fitness_v = calcular_fitness(v)
                    if fitness_v > melhor_fitness_vizinho:
                        melhor_vizinho = v
                        melhor_fitness_vizinho = fitness_v

            # Se não achou vizinho melhor, para a busca
            if melhor_vizinho is None:
                break 
            
            # Se achou, atualiza a solução e continua
            solucao_atual = copy.deepcopy(melhor_vizinho)
            fitness_atual = calcular_fitness(solucao_atual)
            self.historico.append(fitness_atual)

        return solucao_atual, fitness_atual

### ----------------------------------------------------
### PARTE 3: ALGORITMO GENÉTICO (ATIVIDADE 4)
### ----------------------------------------------------

# Parâmetros do AG (definidos no slide)
TAMANHO_POPULACAO = 50
GERACOES_AG = 500
TAXA_MUTACAO = 0.02  # 2%
TAXA_CROSSOVER = 0.8  # 80%
TAMANHO_TORNEIO = 3
N_ELITE = 2

def criar_individuo_ag():
    """Cria um indivíduo (solução) aleatório."""
    return [random.randint(0, 1) for _ in range(N_DIMENSOES)]

def selecao_torneio(populacao_com_fitness):
    """
    Seleção por torneio.
    Sorteia K indivíduos e retorna o melhor deles.
    """
    # Sorteia K indivíduos da população
    participantes = random.sample(populacao_com_fitness, TAMANHO_TORNEIO)
    
    # Ordena os participantes pelo fitness (que está no índice 1)
    participantes.sort(key=lambda item: item[1], reverse=True)
    
    # Retorna o cromossomo do vencedor (que está no índice 0)
    return participantes[0][0]

def aplicar_crossover(pai1, pai2, tipo_crossover):
    """Aplica crossover com 80% de chance."""
    
    # Verifica a taxa de crossover
    if random.random() >= TAXA_CROSSOVER:
        # Se não der a taxa, retorna os próprios pais
        return pai1.copy(), pai2.copy()

    # Se der a taxa, aplica o crossover escolhido
    if tipo_crossover == 'um_ponto':
        ponto = random.randint(1, N_DIMENSOES - 1)
        filho1 = pai1[:ponto] + pai2[ponto:]
        filho2 = pai2[:ponto] + pai1[ponto:]
        
    elif tipo_crossover == 'dois_pontos':
        ponto1 = random.randint(1, N_DIMENSOES - 2)
        ponto2 = random.randint(ponto1 + 1, N_DIMENSOES - 1)
        filho1 = pai1[:ponto1] + pai2[ponto1:ponto2] + pai1[ponto2:]
        filho2 = pai2[:ponto1] + pai1[ponto1:ponto2] + pai2[ponto2:]
            
    elif tipo_crossover == 'uniforme':
        filho1 = []
        filho2 = []
        for i in range(N_DIMENSOES):
            if random.random() < 0.5:
                filho1.append(pai1[i])
                filho2.append(pai2[i])
            else:
                filho1.append(pai2[i])
                filho2.append(pai1[i])
                
    return filho1, filho2

def aplicar_mutacao(individuo):
    """
    Percorre cada gene do indivíduo e tem 2% de chance
    de inverter o bit (mutação bit-flip).
    """
    individuo_mutado = individuo.copy()
    for i in range(N_DIMENSOES):
        if random.random() < TAXA_MUTACAO:
            individuo_mutado[i] = 1 - individuo_mutado[i]
    return individuo_mutado

def executar_ag(tipo_crossover):
    """Roda o Algoritmo Genético completo."""
    
    # 1. Cria a população inicial
    populacao = [criar_individuo_ag() for _ in range(TAMANHO_POPULACAO)]
    
    # Lista para guardar o melhor fitness de cada geração (para o gráfico)
    historico_melhor_fitness = []

    for _ in range(GERACOES_AG):
        # 2. Avalia a população
        # Crio uma lista de tuplas: (cromossomo, fitness)
        populacao_com_fitness = []
        for ind in populacao:
            populacao_com_fitness.append((ind, calcular_fitness(ind)))
        
        # Ordeno para saber quem são os melhores
        populacao_com_fitness.sort(key=lambda item: item[1], reverse=True)
        
        # Salvo o fitness do melhor dessa geração
        historico_melhor_fitness.append(populacao_com_fitness[0][1])
        
        nova_populacao = []
        
        # 3. Elitismo: Os 2 melhores vão direto para a próxima geração
        for i in range(N_ELITE):
            nova_populacao.append(populacao_com_fitness[i][0]) # Pego só o cromossomo

        # 4. Gera o restante da população (Seleção, Crossover, Mutação)
        while len(nova_populacao) < TAMANHO_POPULACAO:
            # Seleciona dois pais
            pai1 = selecao_torneio(populacao_com_fitness)
            pai2 = selecao_torneio(populacao_com_fitness)
            
            # Aplica Crossover
            filho1, filho2 = aplicar_crossover(pai1, pai2, tipo_crossover)
            
            # Aplica Mutação
            filho1 = aplicar_mutacao(filho1)
            filho2 = aplicar_mutacao(filho2)
            
            # Adiciona os novos filhos na população
            nova_populacao.append(filho1)
            # Verifica o tamanho, pois o crossover pode gerar 2 filhos
            if len(nova_populacao) < TAMANHO_POPULACAO:
                nova_populacao.append(filho2)
        
        # A nova geração vira a geração atual
        populacao = nova_populacao
        
    # Fim das gerações: Pega o melhor fitness da última população
    fitness_final = [calcular_fitness(ind) for ind in populacao]
    melhor_fitness_final = max(fitness_final)
    
    return melhor_fitness_final, historico_melhor_fitness

### ----------------------------------------------------
### PARTE 4: EXECUÇÃO DO EXPERIMENTO COMPLETO
### ----------------------------------------------------

# Variáveis globais do problema e experimento
N_DIMENSOES = 20
N_EXECUCOES = 30
MAX_ITERACOES_HC = 200 # Conforme Atividade 3

# O "main" do script
if __name__ == "__main__":
    
    # Dicionário para guardar todos os resultados
    resultados = {}
    
    print("Iniciando experimentos... Isso pode demorar um pouco.")
    start_time = time.time()

    # --- 1. Rodar Hill Climbing (Atividade 3) ---
    print(f"\nRodando Hill Climbing (Tradicional e Estocástico) {N_EXECUCOES} vezes...")
    
    hc_tradicional = HillClimbing(stochastic=False)
    hc_estocastico = HillClimbing(stochastic=True)
    
    lista_resultados_hc_trad = []
    lista_resultados_hc_estoc = []

    for i in range(N_EXECUCOES):
        # Solução inicial aleatória (a mesma para os dois, para ser justo)
        solucao_inicial = [int(random.random() > 0.8) for _ in range(N_DIMENSOES)]
        
        _, fitness_trad = hc_tradicional.executar(solucao_inicial, MAX_ITERACOES_HC)
        lista_resultados_hc_trad.append(fitness_trad)
        
        _, fitness_estoc = hc_estocastico.executar(solucao_inicial, MAX_ITERACOES_HC)
        lista_resultados_hc_estoc.append(fitness_estoc)
        
        print(f"  Simulação HC {i+1}/{N_EXECUCOES} concluída.")

    resultados['HC_Tradicional'] = {"finais": lista_resultados_hc_trad}
    resultados['HC_Estocastico'] = {"finais": lista_resultados_hc_estoc}

    # --- 2. Rodar Algoritmo Genético (Atividade 4) ---
    tipos_crossover_ag = ['um_ponto', 'dois_pontos', 'uniforme']

    for tipo_c in tipos_crossover_ag:
        print(f"\nRodando AG com crossover: {tipo_c} ({N_EXECUCOES} vezes)...")
        
        lista_fitness_finais = []
        lista_convergencia = [] # Lista de listas (30x500)
        
        for i in range(N_EXECUCOES):
            fitness_final, historico_fitness = executar_ag(tipo_c)
            lista_fitness_finais.append(fitness_final)
            lista_convergencia.append(historico_fitness)
            print(f"  Simulação AG-{tipo_c} {i+1}/{N_EXECUCOES} concluída.")
            
        resultados[tipo_c] = {
            "finais": lista_fitness_finais,
            "convergencia": lista_convergencia
        }

    end_time = time.time()
    print(f"\n--- Experimentos concluídos em {end_time - start_time:.1f} segundos ---")

    # --- 3. Gerar Análises e Gráficos ---
    print("\nGerando análises e gráficos...")

    # Tabela de Média e Desvio Padrão
    print("\n--- Tabela: Média e Desvio Padrão (Fitness Final) ---")
    print("-----------------------------------------------------")
    print(f"| Algoritmo         | Média Fitness | Desvio Padrão |")
    print(f"|-------------------|---------------|---------------|")
    
    # Criar dados para o boxplot
    data_para_boxplot = []
    
    for nome_algoritmo, dados in resultados.items():
        media = np.mean(dados["finais"])
        desvio_padrao = np.std(dados["finais"])
        print(f"| {nome_algoritmo:<17} | {media:<13.2f} | {desvio_padrao:<13.2f} |")
        
        # Adiciona os dados para o dataframe do boxplot
        for valor_fitness in dados["finais"]:
            data_para_boxplot.append({
                "Algoritmo": nome_algoritmo.replace('_', ' ').title(),
                "Fitness Final": valor_fitness
            })
    print("-----------------------------------------------------")


    # Gráfico de Convergência (Só dos AGs)
    plt.figure(figsize=(10, 6))
    plt.title("Gráfico de Convergência dos AGs (Média de 30 execuções)")
    plt.xlabel("Geração")
    plt.ylabel("Melhor Fitness (Média)")

    for tipo_c in tipos_crossover_ag:
        # Pega a lista de 30 listas de convergência
        matriz_convergencia = np.array(resultados[tipo_c]["convergencia"])
        # Calcula a média por coluna (por geração)
        media_convergencia = np.mean(matriz_convergencia, axis=0)
        plt.plot(media_convergencia, label=f"AG - {tipo_c}")

    plt.legend()
    plt.grid(True, linestyle='--')
    plt.savefig("grafico_convergencia.png")
    print(f"\nGráfico 'grafico_convergencia.png' salvo.")

    # Gráfico Boxplot (Comparando os 5 algoritmos)
    df_boxplot = pd.DataFrame(data_para_boxplot)

    plt.figure(figsize=(10, 6))
    plt.title("Comparação do Fitness Final (30 execuções)")
    sns.boxplot(data=df_boxplot, x="Algoritmo", y="Fitness Final")
    # Adiciona os pontos por cima para ver a distribuição
    sns.stripplot(data=df_boxplot, x="Algoritmo", y="Fitness Final", color=".3", alpha=0.6) 
    
    plt.ylabel("Fitness Final")
    plt.xlabel("Algoritmo")
    plt.savefig("boxplot_comparativo.png")
    print(f"Gráfico 'boxplot_comparativo.png' salvo.")

    print("\nScript finalizado com sucesso!")