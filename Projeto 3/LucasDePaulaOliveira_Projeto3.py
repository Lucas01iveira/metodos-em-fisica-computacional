import numpy as np 
import matplotlib.pyplot as plt
import imageio
import os

# Defino uma função que cria um sistema 2d de spins numa malha quadrada NxN

def cria_sistema(N):
  # -------------------------------------------------------------------------
  # Parâmetros de entrada:
    # - N: 'tamanho' do sistema

  # Valores de saída:
    # - M: matriz NxN com os spins aleatoriamente distribuídos no sistema
  # -------------------------------------------------------------------------

  M = [] # matriz M a ser preenchida

  for i in range(N): # para cada linha
    vec_spins = [] # define um vetor auxiliar de preenchimento

    for j in range(N): # para cada coluna 
      aux = np.random.uniform(0,1) # sorteia um número entre 0 e 1
      if aux > 1/2: # se for maior que 1/2
        vec_spins.append(+1) # inclui, na coluna 'j', o spin +1
      else: # caso contrário
        vec_spins.append(-1) # inclui, na coluna 'j', o spin -1

    # ao final do preenchimento de cada coluna, acrescenta o vetor vec_spins na linha 'i' de M
    M.append(vec_spins) 

  return M

# Defino uma função que apresenta o sistema de spins em uma representação bidimensional
  
# Obs.: no caso dessa figura a correspondência entre a matriz de spins do sistema e o plot 
# está na ordem correta. (a função imshow já faz isso automaticamente!!!)

def cria_figura(M):

  # -------------------------------------------------------------------------
  # Parâmetros de entrada:
    # - M: matriz do sistema de spins

  # Valores de saída:
    # - Figura quadriculada representando os spins/dipolos up e down do sistema.
    # (quadradinho preto -> spin up   )
    # (quadradinho branco -> spin down)

  # -------------------------------------------------------------------------
  
  ax = plt.figure(figsize=(10,7))
  M_size = len(M) # tamanho do sistema

  # inclui explicitamente as divisões associadas ao "espaço" de cada spin

  for i in range(M_size):
    if i < M_size-1:
      ax = plt.axhline(i+0.5, color='black')
      ax = plt.axvline(i+0.5, color='black')

  # Utiliza a função 'imshow' para apresentar os spins do sistema.
  # Há alguns casos específicos para os quais o 'colormap'
  # deve ser definido de maneira individualizada para manter a coerência
  # da convenção de cores definida

  conta_elementos = 0
  for i in range(M_size):
    for j in range(M_size):
      conta_elementos += M[i][j]

  if conta_elementos == M_size*M_size:
    # se todos os elementos forem iguais a +1, então a contagem deve ser igual à 
    # quantidade de termos da matriz (M_size*M_size)
    ax = plt.imshow(M, cmap='gray') # (cmap = 'gray' já começa na cor preta)

  elif conta_elementos == - M_size*M_size:
    # se todos os elementos forem iguais a -1, então a contagem deve ser igual à 
    # (-) a quantidade de termos da matriz (ou seja, - M_size*M_size)
    ax = plt.imshow(M, cmap='binary') # (cmap = 'binary' já começa na cor branca) 
  
  else:
    # se a contagem não for igual a M_size*M_size nem a M_size*M_size, então
    # quer dizer que o sistema tem uma quantidade variada de spins 

    ax = plt.imshow(M, cmap='binary')
    # nesse caso há termos -1 e +1, então o cmap = 'binary' já dará conta de ajustar
    # as cores adequadamente

  plt.tick_params(left = False, bottom = False, labelleft = False, labelbottom = False)

# defino uma função auxiliar que calcula a variação de energia do flip considerando
# considerando o cálculo manual 

def calcula_DeltaE(eps, h, M, i_flip, j_flip):
  # -------------------------------------------------------------------------

  # Parâmetros de entrada:
    # - M: matriz de spins do sistema
    # - i_flip: linha do spin a ser alterado
    # - j_flip: coluna do spin a ser alterado

  # Valores de saída:
    # - Variação de energia do flip correspondente

  # -------------------------------------------------------------------------
  #print(M)
  #M_size = np.shape(M)[0]

  M_size = len(M) # tamanho do sistema
  
  if i_flip == M_size-1: # verifica se está na última linha 
    if j_flip == M_size-1: # verifica se está no último termo
      Delta_E = 2*M[i_flip][j_flip]*( eps*( M[i_flip][0] + M[i_flip][j_flip-1] + M[i_flip-1][j_flip] + M[0][j_flip]) + h )

    else: # se está na última linha, mas não no último termo,
      Delta_E = 2*M[i_flip][j_flip]*( eps*( M[i_flip][j_flip+1] + M[i_flip][j_flip-1] + M[i_flip-1][j_flip] + M[0][j_flip]) + h )

  else: # se não está na última linha
    if j_flip == M_size-1: # verifica se está na última coluna
      Delta_E = 2*M[i_flip][j_flip]*( eps*( M[i_flip][0] + M[i_flip][j_flip-1] + M[i_flip-1][j_flip] + M[i_flip+1][j_flip]) + h )

    else: # se não está nem na última linha, nem na última coluna, atualiza normalmente
      Delta_E = 2*M[i_flip][j_flip]*( eps*( M[i_flip][j_flip+1] + M[i_flip][j_flip-1] + M[i_flip-1][j_flip] + M[i_flip+1][j_flip]) + h)

  #print(Delta_E)
  # Delta_E = Delta_E * (-1)
  return Delta_E

# Defino uma função que verifica se o spin da posição [i_flip, j_flip] pode ser flipado ou não
# (é aqui que entra a essência do algoritmo de Metrópolis)

def verifica_flip(T, i_flip, j_flip, eps, h, M):
  # -------------------------------------------------------------------------

  # Parâmetros de entrada
    # - T: temperatura
    # - i_flip: índice da linha 
    # - j_flip: índice da coluna
    # - eps: parâmetro de energia do hamiltoniano
    # - h: parâmetro de campo magnético
    # - M: matriz de spins do sistema

  # !!! UNIDADES ESPECIAIS TAIS QUE KB = 1 !!! # 

  # Valores de saída  
    # - True: se o spin do sistema puder ser alterado
    # - False: se o spin do sistema não puder ser alterado

  # -------------------------------------------------------------------------

  # Cálculo da variação de energia associada à alteração de um spin do sistema
  Delta_E = calcula_DeltaE(eps, h, M, i_flip, j_flip)

  if Delta_E < 0: # se a variação de energia for negativa, o flip pode ser efetuado
    return True
  else: # se não
    # faço um sorteio e verifico a probabilidade do flip acontecer
    r = np.random.uniform(0,1)
    prob_flip = np.exp(-Delta_E/T)
    
    if r < prob_flip: # se o número sorteado 'estiver dentro da probabilidade do flip', o flip pode ser efetuado
      return True
    else: # caso contrário, o flip não pode ser efetuado
      return False

def atualiza_Metropolis(T, eps, h, M):
  M_size = len(M)
  aux_linhas = np.arange(0, M_size, 1)
  aux_colunas = np.arange(0, M_size, 1)

  np.random.shuffle(aux_linhas) # embaralha o vetor de linhas
  np.random.shuffle(aux_colunas) # embaralha o vetor de colunas

  for i_flip in aux_linhas:
    for j_flip in aux_colunas:
      if verifica_flip(T, i_flip, j_flip, eps, h, M) == True:
        M[i_flip][j_flip] = (-1)*M[i_flip][j_flip]

# Função teste percorrendo a matriz de spins do sistema com base em vetores de
# linhas/colunas embaralhadas

def atualiza_Metropolis(T, eps, h, M):
  M_size = len(M)
  aux_linhas = np.arange(0, M_size, 1)
  aux_colunas = np.arange(0, M_size, 1)

  np.random.shuffle(aux_linhas) # embaralha o vetor de linhas
  np.random.shuffle(aux_colunas) # embaralha o vetor de colunas

  for i_flip in aux_linhas:
    for j_flip in aux_colunas:
      if verifica_flip(T, i_flip, j_flip, eps, h, M) == True:
        M[i_flip][j_flip] = (-1)*M[i_flip][j_flip]
  
# ____________________________________________________________________________________
# Exemplo de utilização
eps = 1 # parâmetro de energia
h = 0 # parâmetro de campo magnético
N = 10 # dimensão do sistema
T = 1 # temperatura do sistema
Nvar = 100 # número de varreduras / atualizações da simulação

sist = cria_sistema(N) # cria um sistema randômico de spins de dimensão NxN
cria_figura(sist) # constrói uma figura representativa do sistema criado
plt.show() # apresenta a figura

cria_gif(T,eps,h,sist,Nvar,'gif_teste') # simula a evolução do sistema ao longo de Nvar varreduras de atualização
            # e retorna em um gif

# ____________________________________________________________________________________


  