import numpy as np 
import matplotlib.pyplot as plt

# -----------------------------------------------
# 1) Input dos parâmetros iniciais 
'''c=float(input()); 
L=float(input()); 
r = float(input())
dt = float(input())
tf = float(input())
xc = float(input())
k = float(input())'''

# Parâmetros de debug
c = 50 # m/s
L = 1 # m 
r = 1 # adimensional
dt = 0.0001 # s
tf = 0.156 # s
xc = 0.3 # m
k = 1000 # m^-1
dx = c*dt/r # (passo espacial definido dessa forma)

# -----------------------------------------------
# 2) Definição das variáveis relevantes do problema

x = np.arange(0,L+dx, dx) # vetor espacial
t = np.arange(0, tf+dt, dt) # vetor temporal

xsize = len(x) # comprimento do vetor espacial
tsize = len(t) # comprimento do vetor temporal

y = np.zeros( (xsize, tsize) ) # matriz y que representa a função y(x,t)

# -----------------------------------------------
# 3) Aplico as condicões iniciais e as condições de contorno.

# Uma forma de fazer essas aplicações
'''# Condição inicial 
for i in range(xsize):
    y[i][0] = np.exp(-k*(x[i] - xc)**2) # pacote gaussiano

# Condições de contorno
for j in range(tsize): 
    y[0][j] = 0 # borda da esquerda fixa
    y[xsize-1][j] = 0 # borda da direita fixa'''

# Outra forma de fazer essas aplicações
for i in range(xsize):
    for j in range(tsize):
        if x[i] == 0 or x[i] == L:
            y[i][j] = 0
        elif t[j] == 0:
            y[i][j] = np.exp(-k*(x[i] - xc)**2)

#print(y)

# -----------------------------------------------
# 4) Construo a função para atualizar os elementos da matriz y

# Parâmetros da função
# y -> matriz y(x,t)
# x -> vetor de posições do problema
# t -> vetor dos instantes de tempo do problema

def evolucao_onda(y):
    for j in range(tsize-1): # loop para percorrer os tempos 
        for i in range(1,xsize-1): # loop para percorrer as linhas (posições)
            if j == 0:
                y[i][j+1] = 2*(1-r**2)*y[i][j] + (r**2)*(y[i+1][j] + y[i-1][j]) - y[i][0]
            else:
                y[i][j+1] = 2*(1-r**2)*y[i][j] + (r**2)*(y[i+1][j] + y[i-1][j]) - y[i][j-1]
    return y

# Obs.: O "ponto final" do for precisa ter o -1
# para que, ao atualizar o índice seguinte da matriz (i+1 ou j+1),
# não ocorram problemas de "out of bound" nos vetores.
# Lembre-se de que o for "default" sempre começa do 0 e termina 
# no limite estabelecido -1 unidade. 

# Comentário sobre a ordem dos "fors" dentro desse código
'''
Temos que colocar o loop temporal antes do loop espacial. Ou seja,
a lógica correta é: "para cada instante de tempo, atualizamos
a função y em todas as posições", ao invés de: "para cada posição,
atualizamos a função y em todos os tempos". 

Isso porque para atualizar temporalmente a função y em um
dado ponto i (x), precisamos conhecer também os seus valores nos pontos
i-1 (x-dx) e i+1 (x+dx) no instante atual. Contudo, a onda é uma "entidade dinâmica", 
isto é, a amplitude y muda tanto com x quanto com t. Portanto, se a função y
nos pontos i-1 e i+1 não estiver atualizada no instante atual, o cálculo fica incorreto. 

Em outras palavras: colocando o loop espacial primeiro, o que estamos fazendo é atualizando
a função y num dado ponto i para todos os instantes de tempo, mas como essas atualizações 
dependem da função y nos pontos i-1 e i+1 (os quais ainda não foram atualizados pela ordem do loop),
o resultado "diverge" do esperado. 
'''

# -----------------------------------------------
# 5) Apresento os resultados 
# (teste de verificação)

y_xt = evolucao_onda(y)

for ii in range(-30,-1):
     print("{:10.6e}".format( y[xsize//2 + 15 +ii,-1]) )