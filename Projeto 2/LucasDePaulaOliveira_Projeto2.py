import numpy as np
import matplotlib.pyplot as plt

# O campo B será medido em unidades tais que mu_0 = 1
# Começo o problema definindo as principais funções que serão utilizadas

# ________________________________________________________________________________________
# Defino o integrando da componente Bx do campo magnético.
# Parâmetros de entrada:
  # - x: coordenada x do ponto de observação do campo magnético
  # - y: coordenada y do ponto de observação do campo magnético
  # - z: coordenada z do ponto de observação do campo magnético
  # - z_linha: coordenada da espira circular no eixo z
  # - t: parâmetro de integração (descreve posição de um elemento de corrente na espira circular)

# Valor de saída:
  # - Elemento de campo magnético dBx, associado a um elemento de corrente
  # localizada na posição angular t (parâmetro de integração), calculado em (x,y,z)

def dBx(x, y, z, z_linha ,t):
  numerador = a*(z-z_linha)*np.cos(t)
  denominador = ( (x - a*np.cos(t))**2 + (y - a*np.sin(t))**2 + (z-z_linha)**2 )**(3/2)
  const = (1)/(4*np.pi)*I

  return const*(numerador/denominador)

# ________________________________________________________________________________________
# Defino o integrando da componente By do campo magnético.
# Parâmetros de entrada:
  # - x: coordenada x do ponto de observação do campo magnético
  # - y: coordenada y do ponto de observação do campo magnético
  # - z: coordenada z do ponto de observação do campo magnético
  # - z_linha: coordenada da espira circular no eixo z
  # - t: parâmetro de integração (descreve posição de um elemento de corrente na espira circular)

# Valor de saída:
  # - Elemento de campo magnético dBy, associado a um elemento de corrente
  # localizada na posição angular t (parâmetro de integração), calculado em (x,y,z)

def dBy(x, y, z, z_linha, t):
  numerador = a*(z-z_linha)*np.sin(t)
  denominador = ( (x - a*np.cos(t))**2 + (y - a*np.sin(t))**2 + (z-z_linha)**2 )**(3/2)
  const = (1)/(4*np.pi)*I

  return const*(numerador/denominador)

# ________________________________________________________________________________________
# Defino o integrando da componente Bz do campo magnético.
# Parâmetros de entrada:
  # - x: coordenada x do ponto de observação do campo magnético
  # - y: coordenada y do ponto de observação do campo magnético
  # - z: coordenada z do ponto de observação do campo magnético
  # - z_linha: coordenada da espira circular no eixo z
  # - t: parâmetro de integração (descreve posição de um elemento de corrente na espira circular)

# Valor de saída:
  # - Elemento de campo magnético dBz, associado a um elemento de corrente
  # localizada na posição angular t (parâmetro de integração), calculado em (x,y,z)
def dBz(x, y, z, z_linha, t):
  numerador = a*( (y - a*np.sin(t))*np.sin(t) + (x - a*np.cos(t))*np.cos(t) )
  denominador = ( (x - a*np.cos(t))**2 + (y - a*np.sin(t))**2 + (z-z_linha)**2 )**(3/2)
  const = (1)/(4*np.pi)*I

  return -const*(numerador/denominador)

# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
# Defino uma função que pega um elemento de campo magnético dB
# (função de 3 variáveis) e estima a integral numérica do parâmetro
# de integração associado (t) pelo método de simpson no ponto (x,y,z)

# Parâmetros de entrada:
  # - dB: função a ser integrada
  # - x: ponto x de cálculo da integral
  # - y: ponto y de cálculo da integral
  # - z: ponto z de cálculo da integral
  # - z_linha: coordenada da espira circular no eixo z
  # - n: quantidade de partições regulares do intervalo de integração

# Valores de saída:
  # - integr_simpson: aproximação numérica da integral solicitada via método de Simpson

def simpson(dB, x, y, z, z_linha, n):
  b = 2*np.pi
  a = 0

  # defino a lista dos pontos que serão utilizados
  I = (b-a)/n
  pontos = []
  aux = a
  while aux <= b:
    pontos.append(aux)
    aux += I/2
  
  if b not in pontos:
    pontos.append(b)
  
  # Inicio o loop para calcular a integral
  i = 1 # posição do 1º pto médio
  cont = 0 # variável contadora auxiliar
  integr_simpson = 0
  while cont < n:
    cont += 1
    integr_simpson += (1/6)*(dB(x,y,z,z_linha,pontos[i-1]) + 4*dB(x,y,z,z_linha,pontos[i]) + dB(x,y,z,z_linha,pontos[i+1]))*I 
    i += 2
  
  return integr_simpson

# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
# Defino uma função que plota a visualização 3d do campo magnético num grid cúbico com range de (-2,2) em cada direção
# (range escolhido para facilitar a visualização dos vetores)
# Parâmetros de entrada:
  # - a: raio da espira

# Valores de saída:
  # - Plot 3d do campo magnético numa caixa cúbica 4 x 4 x 2

def plota_B_3d(a):
  # _______________________________________________________________________________________
  # Plot 1: Representação 3d do campo 
  ax = plt.figure(figsize=(9,6)).add_subplot(projection='3d')

  vec_malha = np.arange(-2, 2+0.4, 0.4)
  x, y, z = np.meshgrid(
      vec_malha,
      vec_malha,
      np.arange(-1, 1+1, 1)
  )

  Bx = simpson(dBx, x, y, z, z_linha, 10)
  By = simpson(dBy, x, y, z, z_linha, 10)
  Bz = simpson(dBz, x, y, z, z_linha, 10)

  # plot dos vetores
  plt.suptitle('Visualização 3D do campo magnético gerado pela espira', fontsize=18)
  ax.set_xlabel('$x \, [m]$', fontsize=16, rotation = 0)
  ax.set_ylabel('$y \, [m]$', fontsize=16, rotation = -10)
  ax.set_zlabel('$z \, [m]$', fontsize=16, rotation = 0)
  ax.tick_params(labelsize=14)

  ax.quiver(x, y, z, Bx, By, Bz, length=0.3, normalize=True)

  # plot da curva paramétrica da espira
  t = np.arange(0, 2*np.pi+0.1, 0.1)
  x_espira = a*np.cos(t)
  y_espira = a*np.sin(t)
  z_espira = np.zeros(len(t))

  ax.plot(x_espira, y_espira, z_espira, label='Espira circular', color='red', linewidth=3)
  plt.legend(fontsize=14)
  plt.show()
  print()

# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
# Defino uma função que plota a visualização 2d das linhas de campo magnético no plano xz
# num grid simétrico de tamanho lxl

# Parâmetros de entrada:
  # - a: raio da espira
  # - l: comprimento total do grid ('lado do quadrado')
  # - dl: espaçamento entre os pontos do grid 

# Valores de saída:
  # - Plot do perfil das linhas de campo magnético

def plota_B_xz(a, l, dl):
  # _______________________________________________________________________________________
  # Plot 2: Corte no plano xz (vista de perfil)

  vec_malha = np.arange(-l/2, l/2+dl, dl)
  #vec_malha = np.linspace(-l/2, l/2, N)
  x, z = np.meshgrid(
      vec_malha,
      vec_malha
  )

  Bx = simpson(dBx, x, 0, z, z_linha, 10)
  Bz = simpson(dBz, x, 0, z, z_linha, 10) 

  # plot das linhas de campo
  fig, ax = plt.subplots(figsize=(9,6))
  color_array = np.sqrt(Bx**2 + Bz**2)
  strm = ax.streamplot(x, z, Bx, Bz, color = color_array)
  cbar = plt.colorbar(strm.lines)
  cbar.ax.tick_params(labelsize=14)
  cbar.set_label('Módulo do campo magnético $[u.c.]$', rotation=90, fontsize=18)


  # plot da espira
  ax.scatter(-a,0,color='red')
  ax.scatter(a,0,color='red')

  if a > 1: # Se o raio foir maior que 1, um passo de 0.1 é razoável para plotar a espira
    x_aux = np.arange(-a,a+0.1,0.1)
  else: # Se for menor que 1, diminuo a ordem de grandeza para evitar absurdos na imagem
    x_aux = np.arange(-a,a+0.001,0.001)
  
  y_aux = np.zeros(len(x_aux))
  plt.plot(x_aux, y_aux, color='red', linestyle='solid', linewidth=3)

  #ax.set_title('Perfil das linhas de campo geradas pela espira', fontsize=22)
  plt.suptitle('Perfil das linhas de campo geradas pela espira', fontsize=22)
  ax.set_xlabel('$x \,[m]$', fontsize=18)
  ax.set_ylabel('$z \,[m]$', fontsize=18)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)

  plt.show()
  print()

# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
# Defino uma função que constrói um gráfico da componente Bz do campo ao longo de um intervalo simétrico no eixo z 

# Parâmetros de entrada:
  # - a: raio da espira (em m)
  # - z_sim: coordenada z_max para a qual o plot |Bz| x z será feito (|z| < z_max)

# Valores de saída:
  # - Plot |Bz| x z

def plota_Bz(a, z_max):
  # _______________________________________________________________________________________
  # Plot 3: Módulo da componente z ao longo do eixo z (pontos (0,0,z))
  
  # Plot do gráfico
  z = np.arange(-z_max, z_max+0.01, 0.01)
  Bz = simpson(dBz, 0, 0, z, z_linha, 10)

  fig, ax = plt.subplots(figsize=(9,6))

  ax.set_title('Componente Bz ao longo do eixo z', fontsize=22)
  ax.set_ylabel('$B_z(0,0,z) \,\, [u.c.]$', fontsize=18)
  ax.set_xlabel('$z \, [m]$', fontsize=18)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)

  ax.plot(z, Bz, linestyle='solid', color='black')
  ax.axhline(I/(2*a), color='red', linestyle='dashed', label='$I/(2\cdot a) = $' + '{:.2f}'.format(I/(2*a)))

  plt.legend(fontsize=16)
  plt.grid()
  plt.show()

  print()

# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
# Defino uma função que plota o módulo de B ao longo do eixo x 
# Parâmetros de entrada:
  # - a: raio da espira (em m)
  # - xmax: valor máximo da coordenada x a ser apresentado no plot

# Valores de saída:
  # - Gráfico |B(x,0,0)| x (x) no intervalo [0, x_max]
  # - Gráfico |B(x,0,0)| x (x) aproximado no intervalo (a-0.5, a+0.5)

def plota_modB(a, x_max, approach, eps):
  # _______________________________________________________________________________________
  # Plot 4: Módulo do campo ao longo do eixo x ( = módulo da componente z nos pontos (x,0,0))

  # Plot do gráfico
  x = np.arange(0,x_max,0.012)
  Bz = simpson(dBz, x, 0, 0, z_linha, 10)
  B_plot = np.abs(Bz)

  fig, ax = plt.subplots(figsize=(9,6))

  ax.set_title('Módulo do campo magnético ao longo do eixo x', fontsize=22)
  ax.set_ylabel('$|B(x,0,0)| \,\, [u.c.]$', fontsize=18)
  ax.set_xlabel('$z \, [m]$', fontsize=18)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)

  ax.plot(x, B_plot, color='black', linestyle='solid')

  # Plot da posição da espira
  ax.axvline(x=a, color='red', linestyle='dashed', label='a = {}'.format(a))

  plt.legend(fontsize=16)
  plt.grid()
  plt.show()
  print()

  if approach == True:
    # Plot do gráfico com zoom no intervalo próximo à espira:
    x = np.arange(0,x_max,0.012)
    Bz = simpson(dBz, x, 0, 0, z_linha, 10)
    B_plot = np.abs(Bz)

    fig, ax = plt.subplots(figsize=(9,6))

    ax.set_title('Módulo do campo magnético ao longo do eixo x', fontsize=22)
    ax.set_ylabel('$|B(x,0,0)| \,\, [u.c.]$', fontsize=18)
    ax.set_xlabel('$z \, [m]$', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    ax.plot(x, B_plot, color='black', linestyle='solid')

    # Plot da posição da espira
    ax.axvline(x=a, color='red', linestyle='dashed', label='a = {}'.format(a))

    plt.legend(fontsize=16)
    plt.grid()
    plt.xlim(a-eps, a+eps)
    plt.show()
    print()

# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
# Defino uma função que plota a configuração das linhas de campo geradas por um solenóide
# (plano xz)
# Parâmetros de entrada:
  # - l: tamanho do grid de pontos (m)
  # - dl: espaçamento entre os pontos do grid (m)
  # - z_sup: coordenada z (m) limite do solenóide simétrico
  # - dz: espaçamento entre cada espira do solenóide

# Valores de saída:
  # - Plot do perfil das linhas de campo magnético

def plota_solenoide(a, l, dl, z_sup, dz):
  vec_malha = np.arange(-l/2, l/2+dl, dl)

  x, z = np.meshgrid(
      vec_malha,
      vec_malha
  )

  # Defino as matrizes para guardar o campo magnético resultante no espaço
  Bx_resultante = np.zeros( (len(x), len(z)) )
  Bz_resultante = np.zeros( (len(x), len(z)) )

  # Defino a figura do gráfico
  fig, ax = plt.subplots(figsize=(9,6))

  # Defino um vetores auxiliares para fazer o plot das espiras
  z_linha = np.arange(-z_sup, z_sup+dz, dz)
  for i in range(len(z_linha)):
    # Representação das extremidades de cada espira no plano xz
    plt.scatter(-a,z_linha[i], color='red')
    plt.scatter(a,z_linha[i], color='red')

    Bx = simpson(dBx, x, 0, z, z_linha[i], 10)
    Bz = simpson(dBz, x, 0, z, z_linha[i], 10)

    Bx_resultante += Bx
    Bz_resultante += Bz
  
  color_array = np.sqrt(Bx_resultante**2 + Bz_resultante**2)
  strm = ax.streamplot(x, z, Bx_resultante, Bz_resultante, color = color_array)
  cbar = plt.colorbar(strm.lines)
  cbar.ax.tick_params(labelsize=14)
  cbar.set_label('Módulo do campo magnético $[u.c.]$', rotation=90, fontsize=18)

  plt.suptitle('Perfil das linhas de campo geradas pela espira', fontsize=22)
  ax.set_xlabel('$x \,[m]$', fontsize=18)
  ax.set_ylabel('$z \,[m]$', fontsize=18)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)

  plt.show()
  print()

# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
# Exemplo de utilização do código 

# Input dos parâmetros relevantes 
a = 1 # Raio da espira (m)
I = 1 # Corrente (A)
z_linha = 0 # Posição da espira circular no eixo z (m)

l = 4 # comprimento da malha 2d (m)
dl = 0.4 # espaçamento dos pontos da malha 2d (m)

z_max = 3 # coordenada z (m) máxima do plot |Bz(0,0,z)| x z 
x_max = 5 # coordenada x (m) máxima do plot |B(x,0,0)| x (x)  

z_sup = 1 # coordenada z (m) superior do solenóide (simétrico)
dz = 0.3 # espaçamento entre as espiras do solenoide

approach = True # variável auxiliar da função plota_modB
eps = 0.8 # variável auxiliar da função plota_modB

# Chamada das funções 
plota_B_3d(a)
plota_B_xz(a, l, dl)
plota_Bz(a, z_max)
plota_modB(a, x_max, approach, eps)
plota_solenoide(a, l, dl, z_sup, dz)

# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
# Plots do campo de dipolo puro
a = 0.05 # m
I = 1 # A
l = 4 # m
dl = 0.012 # m 

plota_B_xz(a, l, dl)
print()
plota_B_3d(a)
print()

a = 1 # m
I = 1 # A
l = 32 # m
dl = 0.34 # m 

plota_B_xz(a,l,dl)
print()
plota_B_3d(a)
print()

# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
# Comparação entre as expressões analítica e numérica (dipolo puro)

a = 0.01 # m 
I = 1 # A

# Defino uma função para calcular o campo de dipolo analítico em pontos ao longo do eixo x 
def B_dip(x):
  m = I*a**2
  return np.abs( (1/4*np.pi)*(m/x**3) )

# Faço um plot comparativo entre o campo estimado numericamente
x_plot = np.arange(0.5,3,0.013)
B_dip_plot = B_dip(x_plot)

Bz_num = simpson(dBz, x_plot, 0, 0, z_linha, 10) 
# ao longo do eixo x somente a componente Bz do campo é não nula, de modo que |B| = |Bz|

B_plot = np.abs(Bz_num)

plt.figure(figsize=(9,6))
plt.subplot(211)
plt.title('Grafico comparativo entre $|B(x,0,0)|$ e $|B_{dip}(x,0,0)|$', fontsize=22)
plt.plot(x_plot, B_dip_plot, color='red', label='$|B_{dip}(x,0,0)| \,\, [u.c.]$')
plt.plot(x_plot, B_plot, color='black', label='$|B(x,0,0)| \, \, [u.c.]$', linestyle='dashed')
#plt.xlabel('$x \, [m]$', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.legend(fontsize=16)

plt.subplot(212)
delta_B = B_dip_plot - B_plot
plt.plot(x_plot, delta_B, label='Resíduos absolutos', linestyle='dashed', color='black')

y_aux = np.zeros(len(x_plot))
plt.plot(x_plot, y_aux, linestyle='solid', color='red')

plt.xlabel('$x \, [m]$', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(fontsize=16)
plt.show()

# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
# Caso a -> infinito
a = 15
l = 20
dl = 0.137

plota_B_xz(a,l,dl)


# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
# Caso do campo gerado por um solenóide
# Solenóide comum
a = 1 
I = 1 
l = 20 
dl = 0.017
z_sup = 4 
dz = 1

plota_solenoide(a, l, dl, z_sup, dz)

# Solenóide infinito
a = 1
I = 1
l = 50
dl = 0.17
z_sup = 10
dz = 0.1

plota_solenoide(a, l, dl, z_sup, dz)

# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
# Caso do campo gerado por um solenóide
# Solenóide comum
a = 1 
I = 1 
l = 20 
dl = 0.017
z_sup = 4 
dz = 1

plota_solenoide(a, l, dl, z_sup, dz)

# Solenóide infinito
a = 1
I = 1
l = 20
dl = 0.017
z_sup = 10
dz = 0.3

plota_solenoide(a, l, dl, z_sup, dz)