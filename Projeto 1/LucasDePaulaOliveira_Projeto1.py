# Função A (auxiliar) definida nos slides
def A(theta1, theta2, p_theta1, p_theta2, l1, l2, m1, m2):
  import numpy as np 

  return ( p_theta1*p_theta2*np.sin(theta1-theta2) )/( l1*l2*(m1 + m2*(np.sin(theta1-theta2))**2 ) )

# Função B (auxiliar) definida nos slides
def B(theta1, theta2, p_theta1, p_theta2, l1, l2, m1, m2):
  import numpy as np 

  return ( ((l2**2)*m2*(p_theta1**2) + (l1**2)*(m1+m2)*(p_theta2**2) - l1*l2*m2*p_theta1*p_theta2*np.cos(theta1-theta2)) \
          /(2*(l1**2)*(l2**2)*(m1 + m2*(np.sin(theta1-theta2))**2)**2 ) )*np.sin(2*(theta1-theta2))


# Crio uma função que, dados os parâmetros constantes do problema e as variáveis de estudo num determinado
# instante de tempo, a função retorna o valor atualizado dessas variáveis (ou seja,
# o valor delas no instante de tempo seguinte t+dt) considerando o método de evolução temporal
# RK2 (Runge-Kutta ordem 2)


def atualiza(theta1, theta2, p_theta1, p_theta2, l1, l2, m1, m2, g, dt):
  import numpy as np

  # 1) Definem-se os coeficientes k1 para cada uma das variáveis
  k1_theta1 = ( (l2*p_theta1 - l1*p_theta2*np.cos(theta1-theta2) )/( (l1**2)*l2*(m1 + m2*(np.sin(theta1-theta2))**2 ) ) )*dt

  k1_theta2 = ( (- l2*p_theta1*np.cos(theta1-theta2) + l1*(1 + m1/m2)*p_theta2)/( l1*(l2**2)*(m1 + m2*(np.sin(theta1-theta2))**2 ) ) )*dt

  k1_p_theta1 = ( - (m1 + m2)*g*l1*np.sin(theta1) - A(theta1, theta2, p_theta1, p_theta2, l1, l2, m1, m2) \
                 + B(theta1, theta2, p_theta1, p_theta2, l1, l2, m1, m2) )*dt
  
  k1_p_theta2 = ( - m2*g*l2*np.sin(theta2) + A(theta1, theta2, p_theta1, p_theta2, l1, l2, m1, m2) \
                 - B(theta1, theta2, p_theta1, p_theta2, l1, l2, m1, m2))*dt
  
  # 2) Cálculo das variáveis a meio passo

  theta1_12_passo = theta1 + (k1_theta1/2)
  theta2_12_passo = theta2 + (k1_theta2/2)
  p_theta1_12_passo = p_theta1 + (k1_p_theta1/2)
  p_theta2_12_passo = p_theta2 + (k1_p_theta2/2)

  # 3) Definição dos coeficientes k2 para cada uma das variáveis

  k2_theta1 = ( (l2*p_theta1_12_passo - l1*p_theta2_12_passo*np.cos(theta1_12_passo-theta2_12_passo) )\
                        /( (l1**2)*l2*(m1 + m2*(np.sin(theta1_12_passo-theta2_12_passo))**2 ) ) )*dt
  
  k2_theta2 = ( (- l2*p_theta1_12_passo*np.cos(theta1_12_passo-theta2_12_passo) + l1*(1 + m1/m2)*p_theta2_12_passo)\
               /( l1*(l2**2)*(m1 + m2*(np.sin(theta1_12_passo-theta2_12_passo))**2 ) ) )*dt
  
  k2_p_theta1 = ( - (m1 + m2)*g*l1*np.sin(theta1_12_passo) - A(theta1_12_passo, theta2_12_passo, p_theta1_12_passo, p_theta2_12_passo, l1, l2, m1, m2) \
                 + B(theta1_12_passo, theta2_12_passo, p_theta1_12_passo, p_theta2_12_passo, l1, l2, m1, m2) )*dt
  
  k2_p_theta2 = ( - m2*g*l2*np.sin(theta2_12_passo) + A(theta1_12_passo, theta2_12_passo, p_theta1_12_passo, p_theta2_12_passo, l1, l2, m1, m2) \
                 - B(theta1_12_passo, theta2_12_passo, p_theta1_12_passo, p_theta2_12_passo, l1, l2, m1, m2))*dt
  
  # 4) Cálculo das variáveis atualizadas

  theta1_atualizado = theta1 + k2_theta1
  theta2_atualizado = theta2 + k2_theta2

  if theta1_atualizado > np.pi:
    theta1_atualizado = theta1_atualizado - 2*np.pi
  elif theta1_atualizado < - np.pi:
    theta1_atualizado = theta1_atualizado + 2*np.pi
  
  if theta2_atualizado > np.pi:
    theta2_atualizado = theta2_atualizado - 2*np.pi
  elif theta2_atualizado < - np.pi:
    theta2_atualizado = theta2_atualizado + 2*np.pi
  
  p_theta1_atualizado = p_theta1 + k2_p_theta1
  p_theta2_atualizado = p_theta2 + k2_p_theta2

  return theta1_atualizado, theta2_atualizado, p_theta1_atualizado, p_theta2_atualizado

# Crio uma função que, dados os parâmetros e as condições iniciais, retorna 
# os vetores theta1, theta2, p_theta1 e p_theta2 já atualizados temporalmente.

# Parâmetros da função:
# - Constantes do problema (l1, l2, m1, m2, g, dt, Tf)
# - Condições iniciais das variáveis theta1, theta2, p_theta1, p_theta2
# - Time step (dt)
# - Tempo total de estudo do problema (Tf)

def retorna_dados(theta1_0, theta2_0, p_theta1_0, p_theta2_0, l1, l2, m1, m2, g, dt, Tf):
  import numpy as np
  
  # Construção do vetor temporal
  t = np.arange(0, Tf+dt, dt)
  tsize = len(t)

  # Construção dos vetores theta1, theta2, p_theta1, p_theta2 
  # para todo o intervalo de tempo desejado
  theta1 = np.zeros(tsize)
  theta2 = np.zeros(tsize)
  p_theta1 = np.zeros(tsize) 
  p_theta2 = np.zeros(tsize)

  # Aplicação das condições iniciais
  theta1[0] = theta1_0 
  theta2[0] = theta2_0
  p_theta1[0] = p_theta1_0
  p_theta2[0] = p_theta2_0

  for i in range(tsize-1):
    theta1[i+1], theta2[i+1], p_theta1[i+1], p_theta2[i+1] = atualiza(theta1[i], theta2[i], p_theta1[i], p_theta2[i], l1, l2, m1, m2, g, dt)

  return theta1, theta2, p_theta1, p_theta2, t

# Crio uma função que, fornecidos os vetores temporais theta1, theta2, p_theta1,
# p_theta2 (já atualizados) e o vetor t,  retorna os seguintes gráficos:
 
# - theta1 x t
# - theta2 x t
# - (theta1 x t) + (theta2 x t)
# - Delta theta x t
# - p_theta1 x theta1 (órbita da massa 1 no espaço de fases)
# - p_theta2 x theta2  (órbita da massa 2 no espaço de fases)


def apresenta_graficos(theta1, theta2, p_theta1, p_theta2, t):
  import matplotlib.pyplot as plt
  import numpy as np

  # Gráfico 1: theta1 x t
  ax = plt.figure(figsize=(9,6))

  ax = plt.plot(t, theta1, color='blue')

  plt.title('Evolução temporal da coordenada $\u03B8_1$', fontsize=26)
  plt.ylabel('$\u03B8_1$ (rad)', fontsize=24)
  plt.xlabel('t (s)', fontsize=24)

  plt.xticks(fontsize=20)
  plt.yticks(fontsize=20)

  plt.grid()
  plt.show()
  print()

  # Gráfico 2: theta2 x t
  ax = plt.figure(figsize=(9,6))

  ax = plt.plot(t, theta2, color='red')

  plt.title('Evolução temporal da coordenada $\u03B8_2$', fontsize=26)
  plt.ylabel('$\u03B8_2$ (rad)', fontsize=24)
  plt.xlabel('t (s)', fontsize=24)

  plt.xticks(fontsize=20)
  plt.yticks(fontsize=20)

  plt.grid()
  plt.show()
  print()

  # Gráfico 3: theta1 x t e theta2 x t
  ax = plt.figure(figsize=(9,6))

  ax = plt.plot(t, theta1, label='$\u03B8_1 (t)$', color='blue')
  ax = plt.plot(t, theta2, label='$\u03B8_2 (t)$', color='red')

  plt.title('Coordenadas angulares', fontsize=26)
  plt.xlabel('t (s)', fontsize=24)

  plt.xticks(fontsize=20)
  plt.yticks(fontsize=20)
  plt.legend(loc=1, fontsize=18)

  plt.grid()
  plt.show()
  print()


  # Gráfico 4: p_theta1 x theta1
  ax = plt.figure(figsize=(9,6))

  ax = plt.plot(theta1/np.pi, p_theta1, color= 'blue')

  plt.title('Órbita no espaço de fases da massa $m_1$', fontsize=26)
  plt.ylabel('$p_{\u03B81}}$', fontsize=24)
  plt.xlabel('$\u03B8_1 / \pi$', fontsize=24)

  plt.xticks(fontsize=20)
  plt.yticks(fontsize=20)

  plt.grid()
  plt.show()
  print()

  # Gráfico 5: p_theta2 x theta2
  ax = plt.figure(figsize=(9,6))

  ax = plt.plot(theta2/np.pi, p_theta2, color='red')

  plt.title('Órbita no espaço de fases da massa $m_2$', fontsize=26)
  plt.ylabel('$p_{\u03B82}}$', fontsize=24)
  plt.xlabel('$\u03B8_2 / \pi$', fontsize=24)

  plt.xticks(fontsize=20)
  plt.yticks(fontsize=20)

  plt.grid()
  plt.show()

# Crio uma função que, dado o vetor temporal t e os vetores theta e p_theta (já atualizados)
# retorna figura da seção de Poincaré associada.

# Parâmetros:
# - vetor temporal: t
# - theta_plot = variável theta (já atualizada para todos os tempos) para a qual desejamos verificar a seção de Poincaré
# - p_theta_plot = variável p_theta (já atualizada para todos os tempos) para a qual desejamos verificar a seção de Poincaré
# - theta_auxiliar = variável theta auxiliar (necessária para a confecção da figura)

def sec_Poincare(theta_plot, p_theta_plot, theta_auxiliar, dt, Tf):
  import numpy as np 
  import matplotlib.pyplot as plt

  # Defino os vetores necessários
  t = np.arange(0, Tf, dt)
  tsize = len(t)

  thetaSec = []
  p_thetaSec = []

  # Atualizo os vetores thetaSec e p_thetaSec
  for i in range(tsize-1):
    if theta_auxiliar[i] < 0 and theta_auxiliar[i+1] > 0:
      thetaSec.append(theta_plot[i+1])
      p_thetaSec.append(p_theta_plot[i+1])
  
  thetaSec = np.array(thetaSec)
  p_thetaSec = np.array(p_thetaSec)

  ax = plt.figure(figsize=(9,6))

  ax = plt.scatter(thetaSec/np.pi, p_thetaSec, color= 'black')
  plt.title('Seção de Poincaré [\u03B8, $p_{\u03B8}$]', fontsize=26)
  plt.ylabel('$p_{\u03B8}}$', fontsize=24)
  plt.xlabel('$\u03B8 / \pi$', fontsize=24)

  plt.xticks(fontsize=20)
  plt.yticks(fontsize=20)
  plt.grid()

  plt.show()

# ____________________________________________________________________________________________
# 1) Melhor escolha do parâmetro dt
# Caso m1>>m2
from time import time
dt = [1, 0.1, 0.01, 0.001, 0.0001]

theta1_0 = np.pi/2 # rad
theta2_0 = np.pi/2 # rad
p_theta1_0 = 0 # kg*m^2/s
p_theta2_0 = 0 # kg*m^2/s
l1 = 10e-2 # m
l2 = 10e-2 # m 
m1 = 200e-3 # kg 
m2 = 10e-3 # kg
g = 9.8 # m/s^2
Tf = 30 # s

for i in range(len(dt)):
  start = time()
  theta1, theta2, p_theta1, p_theta2, t = retorna_dados(theta1_0, theta2_0, p_theta1_0, p_theta2_0, l1, l2, m1, m2, g, dt[i], Tf)

  print('-'*50)
  print()
  print('Resultados obtidos para o passo temporal dt = {} s'.format(dt[i]))
  print('Tempo necessário para o atualização dos vetores temporais: {:.5f}'.format(time() - start))
  apresenta_graficos(theta1, theta2, p_theta1, p_theta2, t)

# Caso m1 = m2
dt = [1, 0.1, 0.01, 0.001, 0.0001]

theta1_0 = np.pi/2 # rad
theta2_0 = np.pi/2 # rad
p_theta1_0 = 0 # kg*m^2/s
p_theta2_0 = 0 # kg*m^2/s
l1 = 10e-2 # m
l2 = 10e-2 # m 
m1 = 10e-3 # kg 
m2 = 10e-3 # kg
g = 9.8 # m/s^2
Tf = 30 # s

for i in range(len(dt)):
  start = time()
  theta1, theta2, p_theta1, p_theta2, t = retorna_dados(theta1_0, theta2_0, p_theta1_0, p_theta2_0, l1, l2, m1, m2, g, dt[i], Tf)

  print('-'*50)
  print()
  print('Resultados obtidos para o passo temporal dt = {} s'.format(dt[i]))
  print('Tempo necessário para o atualização dos vetores temporais: {:.5f}'.format(time() - start))
  apresenta_graficos(theta1, theta2, p_theta1, p_theta2, t)

# Caso m1 << m2
dt = [1, 0.1, 0.01, 0.001, 0.0001]

theta1_0 = np.pi/2 # rad
theta2_0 = np.pi/2 # rad
p_theta1_0 = 0 # kg*m^2/s
p_theta2_0 = 0 # kg*m^2/s
l1 = 10e-2 # m
l2 = 10e-2 # m 
m1 = 10e-3 # kg 
m2 = 200e-3 # kg
g = 9.8 # m/s^2
Tf = 30 # s

for i in range(len(dt)):
  start = time()
  theta1, theta2, p_theta1, p_theta2, t = retorna_dados(theta1_0, theta2_0, p_theta1_0, p_theta2_0, l1, l2, m1, m2, g, dt[i], Tf)

  print('-'*50)
  print()
  print('Resultados obtidos para o passo temporal dt = {} s'.format(dt[i]))
  print('Tempo necessário para o atualização dos vetores temporais: {:.5f}'.format(time() - start))
  apresenta_graficos(theta1, theta2, p_theta1, p_theta2, t)

# ____________________________________________________________________________________________
# 2) Gráficos do regime m1 >> m2
# Condição inicial 1
theta1_0 = np.pi/2 # rad
theta2_0 = np.pi/2 # rad
p_theta1_0 = 0 # kg*m^2/s
p_theta2_0 = 0 # kg*m^2/s
l1 = 20e-2 # m
l2 = 20e-2 # m 
m1 = 200e-3 # kg 
m2 = 10e-3 # kg
g = 9.8 # m/s^2
dt = 0.001 # s
Tf = 60 # s

theta1, theta2, p_theta1, p_theta2, t = retorna_dados\
(theta1_0, theta2_0, p_theta1_0, p_theta2_0, l1, l2, m1, m2, g, dt, Tf)
apresenta_graficos(theta1, theta2, p_theta1, p_theta2, t)
sec_Poincare(theta1, p_theta1, theta2, dt, Tf)

# Condição inicial 2
theta1_0 = -np.pi/2 # rad
theta2_0 = -np.pi/2 # rad
p_theta1_0 = 0 # kg*m^2/s
p_theta2_0 = 0 # kg*m^2/s
l1 = 20e-2 # m
l2 = 20e-2 # m 
m1 = 200e-3 # kg 
m2 = 10e-3 # kg
g = 9.8 # m/s^2
dt = 0.001 # s
Tf = 60 # s

theta1, theta2, p_theta1, p_theta2, t = retorna_dados\
(theta1_0, theta2_0, p_theta1_0, p_theta2_0, l1, l2, m1, m2, g, dt, Tf)
apresenta_graficos(theta1, theta2, p_theta1, p_theta2, t)
sec_Poincare(theta1, p_theta1, theta2, dt, Tf)

# Plot de delta(theta) (estudo da sensibilidade às condições iniciais)
theta1_0 = np.pi/2 # rad
theta2_0 = 0 # rad
p_theta1_0 = 0 # kg*m^2/s
p_theta2_0 = 0 # kg*m^2/s
l1 = 20e-2 # m
l2 = 20e-2 # m 
m1 = 200e-3 # kg 
m2 = 10e-3 # kg
g = 9.8 # m/s^2
dt = 0.001 # s
Tf = 60 # s

theta1_cond1, theta2_cond1, p_theta1, p_theta2, t = retorna_dados\
(theta1_0, theta2_0, p_theta1_0, p_theta2_0, l1, l2, m1, m2, g, dt, Tf)

theta1_0 = -np.pi/2 # rad
theta2_0 = 0 # rad
p_theta1_0 = 0 # kg*m^2/s
p_theta2_0 = 0 # kg*m^2/s
l1 = 20e-2 # m
l2 = 20e-2 # m 
m1 = 200e-3 # kg 
m2 = 10e-3 # kg
g = 9.8 # m/s^2
dt = 0.001 # s
Tf = 60 # s

theta1_cond2, theta2_cond2, p_theta1, p_theta2, t = retorna_dados\
(theta1_0, theta2_0, p_theta1_0, p_theta2_0, l1, l2, m1, m2, g, dt, Tf)

delta_theta1 = theta1_cond2 - theta1_cond2
delta_theta2 = theta2_cond1 - theta2_cond2
ax = plt.figure(figsize=(9,6))

ax = plt.plot(t, delta_theta1, color='blue', label='$\Delta \u03B8_1 (t)$')
ax = plt.plot(t, delta_theta2, color='red', label='$\Delta \u03B8_2 (t)$')
plt.title('Evolução temporal da variação das coordenadas angulares', fontsize=26)

plt.xlabel('t (s)', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc=1, fontsize=18)

plt.grid()
plt.show()

# ____________________________________________________________________________________________
# 3) Gráficos do caso genérico

theta1_0 = -np.pi/3 # rad
theta2_0 = np.pi/3 # rad
p_theta1_0 = 0 # kg*m^2/s
p_theta2_0 = 0 # kg*m^2/s
l1 = 20e-2 # m
l2 = 20e-2 # m 
m1 = 50e-3 # kg 
m2 = 50e-3 # kg
g = 9.8 # m/s^2
dt = 0.001 # s
Tf = 60 # s

theta1, theta2, p_theta1, p_theta2, t = retorna_dados\
(theta1_0, theta2_0, p_theta1_0, p_theta2_0, l1, l2, m1, m2, g, dt, Tf)
apresenta_graficos(theta1, theta2, p_theta1, p_theta2, t)
sec_Poincare(theta1, p_theta1, theta2, dt, Tf)
print()
sec_Poincare(theta2, p_theta2, theta1, dt, Tf)