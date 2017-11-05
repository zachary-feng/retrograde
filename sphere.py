# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 01:20:10 2017

@author: reemmandil
"""

import os
import numpy 
import matplotlib.pyplot as plt
#%matplotlib inline
#from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
#import matplotlib.tri as mtri
import matplotlib.animation as animation

os.getcwd()
os.listdir(os.getcwd())
data = numpy.loadtxt('stellar_data.csv', delimiter = ',')

#star ID, right ascencsion (in radians), declination (in radians), absolute magnitude 
star_id = data[:,0]
star_ra = data[:,1]
star_dec = data[:,2]
star_mag = data[:,3]

#taking every 50th data point to avoid crowdedness 
minim_ra = star_ra[1::15]
minim_dec = star_dec[1::15]
minim_mag = star_mag[1::15]

#3D celestial sphere of stars
theta_stars = numpy.pi/2 - minim_dec
phi_stars = minim_ra
stars_traj = [numpy.cos(phi_stars)*numpy.sin(theta_stars),numpy.sin(phi_stars)*numpy.sin(theta_stars),numpy.cos(theta_stars)]

#setting the marker size of data points based on absolute magnitude
s = [None]*len(minim_mag)
for i in range(len(minim_mag)): 
 if 0.0 <= minim_mag[i] <=0.1: 
    s[i] = 10
 elif 0.1 < minim_mag[i] and minim_mag[i] <= 0.2:
    s[i] = .1
 elif 0.2 < minim_mag[i] and minim_mag[i] <= 0.3:
    s[i] = .001
 elif 0.3 < minim_mag[i] and minim_mag[i] <= 0.4:
    s[i] = .00001
 elif 0.4 < minim_mag[i] and minim_mag[i] <= 0.5:
    s[i] = .0000001

#plotting the stars' positions with the appropriate magnitude representation
plt.scatter(minim_ra, minim_dec, marker='*', c='yellow', s=s, alpha = 0.4)
            
ax = plt.gca()
ax.set_xlabel('Declination (rad)', fontsize=12) 
ay = plt.gca()
ay.set_ylabel ('Right Ascension (rad)', fontsize=12)

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 10

#mercury data
merc_arr = numpy.load('mercury4.npy')
ra_merc = merc_arr[0]
dec_merc = merc_arr[1]

T_merc = 2112 #hours
t_merc = (-1/T_merc)*numpy.array(range(0,len(ra_merc)))
alpha_merc = numpy.array(numpy.exp(t_merc))

#way to get around the fact that matplotlib does not allow alpha to be anything but a scalar
c_merc = numpy.asarray([(0, 0, 1, a) for a in alpha_merc])

#3D celestial sphere of mercury 
theta_merc = numpy.pi/2 - dec_merc
phi_merc = ra_merc
merc_traj = [numpy.cos(phi_merc)*numpy.sin(theta_merc),numpy.sin(phi_merc)*numpy.sin(theta_merc),numpy.cos(theta_merc)]

#venus data
ven_arr = numpy.load('venus4.npy')
ra_ven = ven_arr[0]
dec_ven = ven_arr[1]

T_ven = 5832 #hours
t_ven = (-1/T_ven)*numpy.array(range(0,len(ra_ven)))
alpha_ven = numpy.array(numpy.exp(t_ven))

#way to get around the fact that matplotlib does not allow alpha to be anything but a scalar
c_ven = numpy.asarray([(0, 0, 1, a) for a in alpha_ven])

#3D celestial sphere of venus
theta_ven = numpy.pi/2 - dec_ven
phi_ven = ra_ven
ven_traj = [numpy.cos(phi_ven)*numpy.sin(theta_ven),numpy.sin(phi_ven)*numpy.sin(theta_ven),numpy.cos(theta_ven)]

#mars data
mars_arr = numpy.load('mars4.npy')
ra_mars = mars_arr[0]
dec_mars = mars_arr[1]

T_mars = 5832 #hours
t_mars = (-1/T_mars)*numpy.array(range(0,len(ra_mars)))
alpha_mars = numpy.array(numpy.exp(t_mars))

#way to get around the fact that matplotlib does not allow alpha to be anything but a scalar
c_mars = numpy.asarray([(0, 0, 1, a) for a in alpha_mars])

#3D celestial sphere of mars
theta_mars = numpy.pi/2 - dec_mars
phi_mars = ra_mars
mars_traj = [numpy.cos(phi_mars)*numpy.sin(theta_mars),numpy.sin(phi_mars)*numpy.sin(theta_mars),numpy.cos(theta_mars)]

#jupiter data
jup_arr = numpy.load('jupiter4.npy')
ra_jup = jup_arr[0]
dec_jup = jup_arr[1]

T_jup = 103894 #hours
t_jup = (-1/T_jup)*numpy.array(range(0,len(ra_jup)))
alpha_jup = numpy.array(numpy.exp(t_jup))

#way to get around the fact that matplotlib does not allow alpha to be anything but a scalar
c_jup = numpy.asarray([(0, 0, 1, a) for a in alpha_jup])

#3D celestial sphere of jupiter
theta_jup = numpy.pi/2 - dec_jup
phi_jup = ra_jup
jup_traj = [numpy.cos(phi_jup)*numpy.sin(theta_jup),numpy.sin(phi_jup)*numpy.sin(theta_jup),numpy.cos(theta_jup)]

#saturn data
sat_arr = numpy.load('saturn4.npy')
ra_sat = sat_arr[0]
dec_sat = sat_arr[1]

T_sat = 258216 #hours
t_sat = (-1/T_sat)*numpy.array(range(0,len(ra_sat)))
alpha_sat = numpy.array(numpy.exp(t_sat))

#way to get around the fact that matplotlib does not allow alpha to be anything but a scalar
c_sat = numpy.asarray([(0, 0, 1, a) for a in alpha_sat])

#3D celestial sphere of saturn
theta_sat = numpy.pi/2 - dec_sat
phi_sat = ra_sat
sat_traj = [numpy.cos(phi_sat)*numpy.sin(theta_sat),numpy.sin(phi_sat)*numpy.sin(theta_sat),numpy.cos(theta_sat)]

#uranus data
ura_arr = numpy.load('uranus4.npy')
ra_ura = ura_arr[0]
dec_ura = ura_arr[1]

T_ura = 736440 #hours
t_ura = (-1/T_ura)*numpy.array(range(0,len(ra_ura)))
alpha_ura = numpy.array(numpy.exp(t_ura))

#way to get around the fact that matplotlib does not allow alpha to be anything but a scalar
c_ura = numpy.asarray([(0, 0, 1, a) for a in alpha_ura])

#3D celestial sphere of uranus
theta_ura = numpy.pi/2 - dec_ura
phi_ura = ra_ura
ura_traj = [numpy.cos(phi_ura)*numpy.sin(theta_ura),numpy.sin(phi_ura)*numpy.sin(theta_ura),numpy.cos(theta_ura)]

#neptune data
nep_arr = numpy.load('neptune4.npy')
ra_nep = nep_arr[0]
dec_nep = nep_arr[1]

T_nep = 1440368 #hours
t_nep = (-1/T_nep)*numpy.array(range(0,len(ra_nep)))
alpha_nep = numpy.array(numpy.exp(t_nep))

#way to get around the fact that matplotlib does not allow alpha to be anything but a scalar
c_nep = numpy.asarray([(0, 0, 1, a) for a in alpha_nep])
                         
#3D celestial sphere of neptune
theta_nep = numpy.pi/2 - dec_nep
phi_nep = ra_nep
nep_traj = [numpy.cos(phi_nep)*numpy.sin(theta_nep),numpy.sin(phi_nep)*numpy.sin(theta_nep),numpy.cos(theta_nep)]

#pluto data
plu_arr = numpy.load('pluto4.npy')
ra_plu = plu_arr[0]
dec_plu = nep_arr[1]

T_plu = 2172480 #hours
t_plu = (-1/T_plu)*numpy.array(range(0,len(ra_plu)))
alpha_plu = numpy.array(numpy.exp(t_plu))

#way to get around the fact that matplotlib does not allow alpha to be anything but a scalar
c_plu = numpy.asarray([(0, 0, 1, a) for a in alpha_plu])
                         
#3D celestial sphere of pluto
theta_plu = numpy.pi/2 - dec_plu
phi_plu = ra_plu
plu_traj = [numpy.cos(phi_plu)*numpy.sin(theta_plu),numpy.sin(phi_plu)*numpy.sin(theta_plu),numpy.cos(theta_plu)]

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

X_1=stars_traj[0]
Y_1=stars_traj[1]
Z_1=stars_traj[2]

X_2=merc_traj[0]
Y_2=merc_traj[1]
Z_2=merc_traj[2]

X_3=ven_traj[0]
Y_3=ven_traj[1]
Z_3=ven_traj[2]

X_4=mars_traj[0]
Y_4=mars_traj[1]
Z_4=mars_traj[2]

X_5=jup_traj[0]
Y_5=jup_traj[1]
Z_5=jup_traj[2]


ax.scatter(X_1,Y_1,Z_1, c='yellow',marker='*', s=s)
#ax.scatter(X_2,Y_2,Z_2, c='green',marker='o', edgecolors = c_merc)
#ax.scatter(X_3,Y_3,Z_3, c='red',marker='o', edgecolors = c_ven)
#ax.scatter(X_4,Y_4,Z_4, c='black',marker='o', edgecolors = c_mars)
#ax.scatter(X_5,Y_5,Z_5, c='orange',marker='o', edgecolors = c_jup)

plt.axis('off')

def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, num-10:num])
        line.set_3d_properties(data[2, num-10:num])
    return lines

# Attaching 3D axis to the figure
#fig = plt.figure()
#ax = p3.Axes3D(fig)
#ax1 = p3.Axes3D(fig)
#
#X = list(X_2)
#Y = list(Y_2)
#Z = list(Z_2)
#
#
#array1 = numpy.ndarray(shape=(3,87), dtype=float)
#
#array1[0] = X
#array1[1] = Y
#array1[2] = Z
#
#
##array2 = numpy.ndarray(shape=(3,181), dtype=float)
#
##array2[0] = X_s
##array2[1] = Y_s
##array2[2] = Z_s
#
## Fifty lines of random 3-D lines
#data = [array1]
##data2 = [Gen_RandLine(25, 3)]            
#   


X2 = list(X_2)
Y2 = list(Y_2)
Z2 = list(Z_2)

X3 = list(X_3)
Y3 = list(Y_3)
Z3 = list(Z_3)

X4 = list(X_4)
Y4 = list(Y_4)
Z4 = list(Z_4)

X5 = list(X_5)
Y5 = list(Y_5)
Z5 = list(Z_5)


# Mars
array1 = numpy.ndarray(shape=(3,876), dtype=float)
array1[0] = X2
array1[1] = Y2
array1[2] = Z2

# Jupiter
array2 = numpy.ndarray(shape=(3,876), dtype=float)
array2[0] = X3
array2[1] = Y3
array2[2] = Z3

# Mercury
array3 = numpy.ndarray(shape=(3,876), dtype=float)
array3[0] = X4
array3[1] = Y4
array3[2] = Z4

# Venus
array4 = numpy.ndarray(shape=(3,876), dtype=float)
array4[0] = X5
array4[1] = Y5
array4[2] = Z5

   
# Fifty lines of random 3-D lines
data = [array1, array2, array3, array4]

# Creating fifty line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]
#lines = [ax.plot(dat[0:1], dat[0:1], dat[0:1])[0] for dat in data]

        
#print len(X)

#
#Setting the axes properties
ax.set_xlim3d([-1.0, 1.0])
ax.set_xlabel('X')

ax.set_ylim3d([-1.0, 1.0])
ax.set_ylabel('Y')

ax.set_zlim3d([-1.0, 1.0])
ax.set_zlabel('Z')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, 876, fargs=(data, lines),
                                   interval=50, blit=False)
#line_ani.save('sphere.mp4', writer = "ffmpeg")
ax.set_facecolor('black')
plt.show()
