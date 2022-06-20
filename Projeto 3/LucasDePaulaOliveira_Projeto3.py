import numpy as np 
import matplotlib.pyplot as plt
import imageio

# ____________________________________________________________________________________
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

# ____________________________________________________________________________________
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

# ____________________________________________________________________________________
# defino uma função auxiliar que calcula a variação de energia do flip considerando
# o cálculo manual 

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

# ____________________________________________________________________________________
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

# ____________________________________________________________________________________
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

# Defino uma função para calcular a magnetização do sistema

# ____________________________________________________________________________________
def calcula_Mag(M):
  # -------------------------------------------------------------------------
  # Parâmetros de entrada:
    # - M: matriz de spins do sistema

  # Valores de saída:
    # - magnetização média do sistema (em módulo)
  # -------------------------------------------------------------------------

  #return np.abs(np.mean(M))
  return np.mean(M)

# ____________________________________________________________________________________
# Defino uma função para criar um gif mostrando a 'evolução temporal' de um determinado sistema de spins

def cria_gif(T, eps, h, M, N_frames, Name):
  # -------------------------------------------------------------------------
  # Parâmetros de entrada
    # - T: temperatura de equilíbrio do sistema
    # - eps: constante de energia 
    # - h: parâmetro de campo magnético
    # - M: matriz do sistema inicial 
    # - N_frames: número de frames desejado
    # - Name: (str) nome do arquivo.gif

  # Valores de saída:
    # - Gif de evolução do sistema a cada iteração sobre o conjunto de spins
  # -------------------------------------------------------------------------
  M_size = len(M) # tamanho do sistema
  filenames = [] # vetor auxiliar para guardar o nome das figuras que compõem cada frame do gif
  ax = plt.figure(figsize=(9,6))
  
  for a in range(N_frames): # para cada frame
    for i in range(5):
      # Constrói a figura a ser apresentada (adaptado da função plota_sistema)

      # ____________________________________________________________________________________
      for i in range(M_size):
        if i < M_size-1:
          ax = plt.axhline(i+0.5, color='black')
          ax = plt.axvline(i+0.5, color='black')

      conta_elementos = 0
      for l in range(M_size):
        for c in range(M_size):
          conta_elementos += M[l][c]

      if conta_elementos == M_size*M_size:
        ax = plt.imshow(M, cmap='gray') # (o cmap = 'gray' já começa na cor preta)
      elif conta_elementos == - M_size*M_size:
        ax = plt.imshow(M, cmap='binary') 
      else:
        ax = plt.imshow(M, cmap='binary')

      plt.tick_params(left = False, bottom = False, labelleft = False, labelbottom = False)
      # ____________________________________________________________________________________

      # construção dos frames do gif
      filename = 'frame_{}.png'.format(a)
      filenames.append(filename)

      if a == N_frames-1: # se estiver no último frame do gif 
        for k in range(5):
          filenames.append(filename) # acrescenta uma pausa maior

      plt.savefig('{}'.format(filename)) # salvo as figuras dentro do diretório criado
      plt.close()

    # Atualiza o sistema para construir o frame seguinte 
    if a < N_frames-1:
      atualiza_Metropolis(T, eps, h, M)
      #TESTE_atualiza_Metropolis(T, eps, h, M)

  # Cria o gif 
  with imageio.get_writer('{}.gif'.format(Name), mode='I') as writer:
    for filename in filenames:
      image = imageio.imread('{}'.format(filename))
      writer.append_data(image)

  
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
# Plot da solução de Onsager
T_plot = np.arange(0.5,3,0.02)
eps = 1 # eps1 = eps2 = eps (assumindo isotropia)
m_plot = (1 - (np.sinh(2*eps/T_plot) * np.sinh(2*eps/T_plot))**(-2) )**(1/8)

ax = plt.figure(figsize=(9,6))
plt.title('Curva de magnetização proposta por Onsager', fontsize=20)
plt.ylabel(r'|$\langle m \rangle$|', fontsize=18)
plt.yticks(fontsize=16)
plt.xlabel(r'$T \,[ u.c.]$', fontsize=18)
plt.xticks(fontsize=16)

plt.plot(T_plot, m_plot, linestyle='solid', color='black')
plt.axvline(2/(np.log(1+np.sqrt(2))), linestyle='dashed', color='red', label=r'$T_c \approx 2.27$')
plt.legend(fontsize=14)

plt.show()

# ____________________________________________________________________________________
# Caso h = 0 e epsilon = 1 

# Avaliação inicial do sistema 
'''
sist = cria_sistema(10) # cria um sistema inicial
eps = 1 # constante de energia 
h = 0 # campo magnético desligado
Nvar = 100 # quantidade de iterações por simulação
T = [1, 2, 3, 6, 9] # vetor de temperaturas

print('Sistema inicial')
cria_figura(sist)
plt.show()
print()
print('-'*30)
for i in range(len(T)):
  sist_aux = np.copy(sist)

  cria_gif(T[i], eps, h, sist_aux, Nvar, 'T={}'.format(T[i]))
  # lembrando que a função cria_gif já atualiza diretamente a matriz fornecida 
  # (então de fato a sequência de frames apresentada no gif é a sequência de atualizações
  # pela qual o sistema mesmo!)

  #for j in range(Nvar):
  #  atualiza_Metropolis(T[i], eps, h, sist_aux)

  print('T = {}'.format(T[i]))
  cria_figura(sist_aux)
  plt.show()
  print()
'''

# Atenção! A sequência de códigos acima foi executada uma única vez para 
# salvar as configurações / gifs e utilizar no relatório. 

# Se executada novamente, serão obtidos novos sistemas de spins e sequências de atualização
# distintas uma vez que o processo de flip de spins é randômico / probabilístico.

# Plot de <m> x T
T = np.arange(0.5,6,0.1) # vetor de temperaturas
eps = 1 # parâmetro de energia
h = 0 # parâmetro de campo magnético
Nvar = 1000 # número de varreduras
mag_mean = [] # vetor para armazenar a magnetização média final obtida para cada temperatura
sist = cria_sistema(10) # define um sistema de spins 10x10
#plota_sistema(sist)
#plt.show()

for i in range(len(T)): # para cada temperatura
  sist_aux = np.copy(sist) # copia o sistema inicial de spins criado
  #sist_aux = cria_sistema(15)
  mag = [] # vetor para armazenar a magnetização do sistema para cada varredura

  mag.append(calcula_Mag(sist_aux)) # calcula a magnetização inicial do sistema criado
  for j in range(Nvar): # para cada varredura
    atualiza_Metropolis(T[i], eps, h, sist_aux) # atualiza o sistema
    #TESTE_atualiza_Metropolis(T[i], eps, h, sist_aux) # atualiza o sistema
    mag.append(calcula_Mag(sist_aux)) # calcula a magnetização e guarda

  mag_mean.append(np.abs(np.mean(mag))) # armazena a magnetização média do sistema

# Apresenta o gráfico correspondente
ax = plt.figure(figsize=(9,6))
ax = plt.scatter(T, mag_mean, color='black')
plt.title('Magnetização média por sítio em função da temperatura ', fontsize=20)
plt.ylabel(r'|$\langle m \rangle|$', fontsize=18)
plt.xlabel('$T$ [u.c.]', fontsize=18)
plt.axvline(x = 2.27, color='red',linestyle='dashed', label=r'$T_c \approx 2.27$')
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.ylim(0,1.5)
plt.show()

# Análise do problema da metaestabilidade
'''A = cria_sistema(10)
cria_figura(A)

for i in range(1000):
  atualiza_Metropolis(0.2,1,0,A)

cria_figura(A)
print(calcula_mag(A))
'''
# Sequência de códigos executada uma única vez 


# ____________________________________________________________________________________
# Efeitos do campo magnético
# (Mesmo sistema / parâmetros)
T = np.arange(0.5,6,0.1) # vetor de temperaturas
eps = 1 # parâmetro de energia
Nvar = 1000 # número de varreduras (pela metade para ganhar +tempo)
h = np.arange(0,3,1) # vetor de parâmetros do campo magnético

ax = plt.figure(figsize=(9,6))
markers = ['o', '^', '*']
colors = ['blue', 'orange', 'green']


for i in range (len(h)): # para cada tamanho de sistema
  sist = cria_sistema(10) # define um sistema randômico de spins
  mag_mean = [] # vetor para armazenar a magnetização media final obtida para cada temperatura

  for j in range(len(T)): # para cada temperatura
    sist_aux = np.copy(sist) # cria uma cópia do sistema formado
    mag = [] # vetor para armazenar a magnétização por sítio de cada iteração

    mag.append(calcula_Mag(sist_aux)) # calcula a primeira magnetização por sítio
    
    for k in range(Nvar): # para cada iteração
      atualiza_Metropolis(T[j], eps, h[i], sist_aux) # atualiza a matriz de spins
      mag.append(calcula_Mag(sist_aux)) # calcula a magnetização por sítio
    
    mag_mean.append(np.abs(np.mean(mag))) # calcula a magnetização média total para a temperatura em questão
  
  ax = plt.scatter(T, mag_mean, linestyle='solid', marker=markers[i], label = 'h = {}'.format(h[i]) , color=colors[i], linewidth = 1)

plt.title('Magnetização média por sítio em função da temperatura ', fontsize=20)
plt.ylabel(r'$\langle |m| \rangle$', fontsize=18)
plt.xlabel('$T$ [u.c.]', fontsize=18)
plt.axvline(x = 2.27, color='red',linestyle='dashed', label=r'$T_c \approx 2.27$')
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

#mag_mean = [] # vetor para armazenar a magnetização média final obtida para cada temperatura
#sist = cria_sistema(10) # define um sistema de spins 50x50

# ____________________________________________________________________________________
# Análise complementar: susceptibilidade magnética
# Defino uma função para calcular a derivada numérica

def derivada(fx, x, dx):
  # ------------------------------------------------------------------------
  # Parâmetros de entrada:
    # - fx: vetor de valores de uma determinada função f calculada nos pontos x
    # - x: vetor de valores nos quais a função f está sendo avaliada
    # - dx: passo entre os pontos do vetor x

  # Valores de saída:
    # - df: vetor de derivadas numéricas da f calculadas em cada ponto x do vetor

  # ------------------------------------------------------------------------
  df = [] # vetor para guardar a derivada de f 

  for i in range(len(x)): # para cada ponto no vetor x

    if i == 0: # se estiver na extremidade esquerda
      deriv = (fx[i+1] - fx[i])/dx # derivada lateral esquerda

    elif i == len(x)-1: # se estiver na extremidade direita
      deriv = (fx[i] - fx[i-1])/dx # derivada lateral direita

    else: # caso contrário
      deriv = (fx[i+1] - fx[i-1])/(2*dx) # derivada simétrica
    
    df.append(deriv)
  return df

# Plot de <m> x h

sist = cria_sistema(10) # define um sistema 10x10
eps = 1 # parâmetro de energia
T = [1,3,5,10] # vetor de temperaturas do sistema
dh = 0.05 # passo de campo magnético 
h = np.arange(-3, 3+dh, dh) # vetor de parâmetros de campo magnético
colors=['blue', 'green', 'red', 'orange'] # vetor de cores
markers=['s', '*', '^', 'o'] # vetor de marcadores
Nvar = 1000 # número de iterações a cada cálculo de <m>

ax = plt.figure(figsize=(9,6))

for i in range(len(T)): # para cada temperatura
  mag_mean = [] # vetor para guardar a média das magnetizações

  for j in range(len(h)): # para cada campo magnético
    sist_aux = np.copy(sist) # copia o sistema inicial de spins criado
    mag = [] # vetor para armazenar a magnetização do sistema para cada varredura

    mag.append(calcula_Mag(sist_aux)) # calcula a magnetização por sítico inicial do sistema criado
    for k in range(Nvar): # para cada varredura
      atualiza_Metropolis(T[i], eps, h[j], sist_aux) # atualiza o sistema
      mag.append(calcula_Mag(sist_aux)) # calcula a magnetização por sítio e guarda

    mag_mean.append(np.mean(mag)) # armazena a magnetização média do sistema

  # Apresenta o gráfico correspondente
  ax = plt.plot(h, mag_mean, label=r'$T = {}$'.format(T[i]), color=colors[i], marker=markers[i])


plt.title('Magnetização média por sítio em função do campo magnético ', fontsize=20)
plt.ylabel(r'$\langle m \rangle$', fontsize=18)
plt.xlabel('$h$ [u.c.]', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=14)
#plt.ylim(-1,1)
plt.show()

# curva de susceptibilidade magnética
sist = cria_sistema(10) # define um sistema 10x10
eps = 1 # parâmetro de energia
T = [1,3,5,10] # vetor de temperaturas do sistema
dh = 0.05 # passo de campo magnético 
h = np.arange(-2, 2+dh, dh) # vetor de parâmetros de campo magnético
colors=['blue', 'green', 'red', 'orange'] # vetor de cores
markers=['s', '*', '^', 'o'] # vetor de marcadores
Nvar = 1000 # número de iterações a cada cálculo de <m>

ax = plt.figure(figsize=(9,6))

for i in range(len(T)): # para cada temperatura
  mag_mean = [] # vetor para guardar a média das magnetizações

  for j in range(len(h)): # para cada campo magnético
    sist_aux = np.copy(sist) # copia o sistema inicial de spins criado
    mag = [] # vetor para armazenar a magnetização do sistema para cada varredura

    mag.append(calcula_Mag(sist_aux)) # calcula a magnetização por sítico inicial do sistema criado
    for k in range(Nvar): # para cada varredura
      TESTE_atualiza_Metropolis(T[i], eps, h[j], sist_aux) # atualiza o sistema
      mag.append(calcula_Mag(sist_aux)) # calcula a magnetização por sítio e guarda

    mag_mean.append(np.mean(mag)) # armazena a magnetização média do sistema
  x_m = derivada(mag_mean, h, dh)

  # Apresenta o gráfico correspondente
  if i > 0:
    ax = plt.plot(h, x_m, label=r'$T = {}$'.format(T[i]), color=colors[i], marker=markers[i])

plt.title('Susceptibilidade magnética', fontsize=20)
plt.ylabel(r'$\chi_M$', fontsize=18)
plt.xlabel('$h$ [u.c.]', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=14)
plt.show()

# ____________________________________________________________________________________

# ____________________________________________________________________________________

# ____________________________________________________________________________________


  
