import math 
import matplotlib.pyplot as plt 
import numpy as np 

# Définition du facteur de distorsion relativiste delta 
def facteur_delta(beta, theta):
    gamma = 1/math.sqrt(1 - beta**2)
    return 1/(gamma*(1 - beta*math.cos(theta)))


# On trace la courbe de delta en fonction de theta 
theta = list(np.linspace(-math.pi, math.pi, 100))
plt.plot(theta, [facteur_delta(0.999,theta[k]) for k in range(100)]) 
plt.xlabel("angle d'incidence de l'observateur")
plt.ylabel("distorsion relativiste")
plt.show()

# On calcule l'aire totale sous la courbe de delta 
tab = np.array([facteur_delta(0.999,theta[k]) for k in range(100)])
print(f'aire totale sous la courbe {np.trapz(tab, np.array(theta), 0.01)}')

# On calcule l'aire sous la courbe de delta entre -1/gamma et 1/gamma 
gamma = 1/math.sqrt(1 - 0.999**2)
theta_prim = np.array(list(np.linspace(-1/gamma, 1/gamma, 100)))
tab_prim = np.array([facteur_delta(0.999,theta_prim[k]) for k in range(100)])
print(f'aire principale : {np.trapz(tab_prim, np.array(theta_prim), 0.01)}')
print(f'angle de coupure : {1/gamma}')

