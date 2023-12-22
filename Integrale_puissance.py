import numpy as np
from scipy.integrate import quad, dblquad, tplquad
import matplotlib.pyplot as plt
from scipy import stats

r_T = 3.09 * 10 ** 15 # B. Cerutti and G. Giacinti 2021
gamma_T = 3*10**6 # Kennel and Coroniti
q = 1.5
p = 2
R_cy = 9.46*10**15 * 6523 * 0.1 * 1 / 3600 * np.pi / 180
    
print(f"R_cy : {R_cy:e}")
print(f"r_t : {r_T:e}")

theta_obs = 24 * np.pi / 180
        
theta_W = 10 * np.pi/180       

def u(alpha) : 
    N = np.sqrt(1 + np.tan(alpha)**2 + np.tan(theta_obs)**2)
    return 1/N, np.tan(alpha)/N, np.tan(theta_obs)/N

def integrand(r, theta, phi, nu, alpha):
    gamma = gamma_T*(r/r_T)**q
    beta = np.sqrt(1-1/gamma**2)
    x,y,z = u(alpha)
    cos_theta_d = x*np.sin(theta)*np.cos(phi) + y*np.sin(theta)*np.sin(phi)+z*np.cos(theta)

    return nu**((1-p)/2)*(1/(gamma*(1-beta*cos_theta_d)))**((5+p)/2)*r**2*np.sin(theta)

def condition_c(r, theta, phi, d_cy, alpha):
    x_M = r*np.sin(theta)*np.cos(phi)
    y_M = r*np.sin(theta)*np.sin(phi)
    z_M = r*np.cos(theta)
    x,y,z = u(alpha)

    return (x_M - d_cy)**2 + y_M**2 + z_M**2 - (x*(x_M-d_cy) + y*y_M + z*z_M)**2 <= R_cy**2

def integrate_over_volume(r_T, theta_W, nu, d_cy, alpha):

    N = np.sqrt(1 + np.tan(alpha)**2 + np.tan(theta_obs)**2)
    t1 = -d_cy*np.sin(theta_W)*N/(np.sin(theta_W)-np.tan(theta_obs)*np.cos(theta_W))
    t2 = -d_cy*np.sin(theta_W)*N/(np.sin(theta_W)+np.tan(theta_obs)*np.cos(theta_W))

    mint = min(t1,t2)
    maxt = max(t1,t2)

    r1 = (mint/N + d_cy)/np.cos(theta_W)-R_cy/np.sin(theta_obs)
    r2 = (maxt/N + d_cy)/np.cos(theta_W)+R_cy/np.sin(theta_obs)


    y1 = np.tan(alpha)*mint/N
    y2 = np.tan(alpha)*maxt/N
    x1 = (mint/N + d_cy)
    x2 = (maxt/N + d_cy)

    if 0 <= alpha <= np.pi/2:
        phi1 = np.arctan(y1/x1)
        phi2 = np.arctan(y2/x2)
    elif  np.pi/2 <= alpha <= 3*np.pi/2 :
        phi1 = np.arctan(y1/x1) + np.pi
        phi2 = np.arctan(y2/x2) + np.pi
    else :
        phi1 = np.arctan(y1/x1) + 2 * np.pi
        phi2 = np.arctan(y2/x2) + 2 * np.pi
    
    if 0 <= alpha <= np.pi/2 or np.pi <= alpha <= 3*np.pi/2:
        phi_min = min(phi1,phi2) - R_cy/max(r1,r2)
        phi_max = max(phi1,phi2) + R_cy/min(r1,r2)
    else :
        phi_min = min(phi1,phi2) - R_cy/min(r1,r2)
        phi_max = max(phi1,phi2) + R_cy/max(r1,r2)

    #print(f"r1 : {r1:e} et r2 : {r2:e}")
    #print(f"Bornes d'integration : r in [{max(r_T/(gamma_T**(1/q)),r1):e}, {min(r_T,r2):e}], phi in [{phi_min:e}, {phi_max:e}], theta in [{np.pi/2 - theta_W:e}, {np.pi/2 + theta_W:e}]" )

    result, error = tplquad(
        lambda r, theta, phi: integrand(r, theta, phi, nu, alpha) if condition_c(r, theta, phi, d_cy, alpha) else 0,
        phi_min, phi_max,
        np.pi/2 - theta_W, np.pi/2 + theta_W,
        max(r_T/(gamma_T**(1/q)),r1), min(r_T,r2),
        epsabs=1, epsrel=1
    )

    return result


def Puissance_recue(nu, alpha, d_cy, nb_points):
    # Variation de d_cy
    d_cy_values = np.linspace(r_T/30, r_T/20, nb_points)
    d_cy_axis = [d/r_T for d in d_cy_values]
    Y_d_cy = [integrate_over_volume(r_T, theta_W, nu, x, alpha) for x in d_cy_values]

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.plot(d_cy_axis, Y_d_cy, marker='o')
    plt.title(f"Variation de d_cy \n Paramètres: nu = {nu:.2e}, alpha = {alpha:.2e}")
    plt.xlabel('d_cy/r_T')
    plt.ylabel('Puissance reçue')

    # Variation de nu
    nu_values = np.linspace(1, 100, nb_points)
    Y_nu = [integrate_over_volume(r_T, theta_W, x, d_cy, alpha) for x in nu_values]

    plt.subplot(132)
    plt.plot(nu_values, Y_nu, marker='o')
    plt.title(f"Variation de nu \n Paramètres: d_cy = {d_cy:.2e}, alpha = {alpha:.2e}")
    plt.xlabel('nu')
    plt.ylabel('Puissance reçue')

    # Variation d'alpha
    alpha_values = np.linspace(0, np.pi, nb_points)
    Y_alpha = [integrate_over_volume(r_T, theta_W, nu, d_cy, x) for x in alpha_values]

    plt.subplot(133)
    plt.plot(alpha_values, Y_alpha, marker='o')
    plt.title(f"Variation de alpha \n Paramètres: d_cy = {d_cy:.2e}, nu = {nu:.2e}")
    plt.xlabel('alpha')
    plt.ylabel('Puissance reçue')

    plt.suptitle("Puissance reçue en fixant 2 paramètres sur 3")
    plt.tight_layout()
    plt.show()

Puissance_recue(1, 0, r_T/10, 10)

