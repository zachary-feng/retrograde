#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 10:28:29 2017

@author: Nathan Leitao/Reem Madil/Zachary Feng/Rane Simpson/Soud Kharusi
"""


import os
import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
theta0 = 0
phi0 = 0
os.getcwd()
os.listdir(os.getcwd())
data = numpy.loadtxt('copy_stellar_data.csv', delimiter=',')


#projection map

def thetaObserver(theta,theta0):
   #if theta -theta0< numpy.pi/2:
   return theta-theta0
#    #else:
#       return theta -theta0- numpy.pi/2




f = lambda R, theta, phi, phi0: [(R/2)*numpy.sin(theta)/(numpy.cos((theta)/2))**2,phi-phi0]

def projection(R,theta,theta0,phi,phi0):
   '''Project a point in the sky prescribed by polar angles (theta, phi) relative to the celestial north pole, onto a disk of radius R.
   The function will return this point in radial coordinates (r,phi)'''
   #if numpy.abs(theta - theta0) <= numpy.pi/2:
   return f(R,theta-theta0,phi,phi0)
   #else: return None


# star ID, right ascencsion (in radians), declination (in radians), absolute magnitude
star_id = data[:, 0]
star_ra = data[:, 1]
star_dec = data[:, 2]
star_mag = data[:, 3]

# taking every 50th data point to avoid crowdedness
minim_ra = star_ra[1::50]
minim_dec = star_dec[1::50]
minim_mag = star_mag[1::50]

# 3D celestial sphere of stars
theta_stars =  [thetaObserver(theta,theta0) for theta in minim_dec]
phi_stars = minim_ra - numpy.pi
stars_traj = [numpy.cos(phi_stars) * numpy.sin(theta_stars), numpy.sin(phi_stars) * numpy.sin(theta_stars),
             numpy.cos(theta_stars)]

#returns a list of the cooridinates on the disk [[r1,r2,...],[phi1,phi2,...]] for the available stars in the relevant hemisphere
def filterForSterProjection(theta_stars, phi_stars,theta0,phi0):
   toReturn = []
   for n in range(0, len(theta_stars)):
       if (numpy.abs(theta_stars[n]-theta0)<=numpy.pi/2 and numpy.abs(phi_stars[n]-phi0)<=numpy.pi/2):
           toReturn.append(projection(1,theta_stars[n],theta0,phi_stars[n],phi0))
   return list(map(list, zip(*toReturn)))


stars_stereograph = filterForSterProjection(theta_stars,phi_stars,theta0,phi0)
print(stars_stereograph)
print(len(stars_stereograph[0])/len(theta_stars))

# setting the marker size of data points based on absolute magnitude
s = [None] * len(minim_mag)
for i in range(len(minim_mag)):
   if 0.0 <= minim_mag[i] <= 0.1:
       s[i] = 10
   elif 0.1 < minim_mag[i] and minim_mag[i] <= 0.2:
       s[i] = 20
   elif 0.2 < minim_mag[i] and minim_mag[i] <= 0.3:
       s[i] = 30
   elif 0.3 < minim_mag[i] and minim_mag[i] <= 0.4:
       s[i] = 40
   elif 0.4 < minim_mag[i] and minim_mag[i] <= 0.5:
       s[i] = 50




# plotting the stars' positions with the appropriate magnitude representation
plt.scatter(theta_stars, phi_stars, marker='*', c='yellow', s=s)

ax = plt.gca()
ax.set_xlabel('Declination (rad)', fontsize=12)
ay = plt.gca()
ay.set_ylabel('Right Ascension (rad)', fontsize=12)



fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 10



# mercury data
merc_arr = numpy.load('10_Years/mercury4.npy')
ra_merc = merc_arr[0]
dec_merc = merc_arr[1]

T_merc = 2112  # hours
t_merc = (-1 / T_merc) * numpy.array(range(0, len(ra_merc)))
alpha_merc = numpy.array(numpy.exp(t_merc))

# way to get around the fact that matplotlib does not allow alpha to be anything but a scalar
c_merc = numpy.asarray([(0, 0, 1, a) for a in alpha_merc])

# 3D celestial sphere of mercury
theta_merc = numpy.pi / 2 - dec_merc
phi_merc = ra_merc
merc_traj = [numpy.cos(phi_merc) * numpy.sin(theta_merc), numpy.sin(phi_merc) * numpy.sin(theta_merc),
            numpy.cos(theta_merc)]

merc_ster_traj = filterForSterProjection(theta_merc,phi_merc,theta0,phi0)



# venus data
ven_arr = numpy.load('10_Years/venus4.npy')
ra_ven = ven_arr[0]
dec_ven = ven_arr[1]

T_ven = 5832  # hours
t_ven = (-1 / T_ven) * numpy.array(range(0, len(ra_ven)))
alpha_ven = numpy.array(numpy.exp(t_ven))

# way to get around the fact that matplotlib does not allow alpha to be anything but a scalar
c_ven = numpy.asarray([(0, 0, 1, a) for a in alpha_ven])

# 3D celestial sphere of venus
theta_ven = numpy.pi / 2 - dec_ven
phi_ven = ra_ven
ven_traj = [numpy.cos(phi_ven) * numpy.sin(theta_ven), numpy.sin(phi_ven) * numpy.sin(theta_ven), numpy.cos(theta_ven)]

ven_ster_traj = filterForSterProjection(theta_ven,phi_ven,theta0,phi0)



# mars data
mars_arr = numpy.load('10_Years/mars4.npy')
ra_mars = mars_arr[0]
dec_mars = mars_arr[1]

T_mars = 5832  # hours
t_mars = (-1 / T_mars) * numpy.array(range(0, len(ra_mars)))
alpha_mars = numpy.array(numpy.exp(t_mars))

# way to get around the fact that matplotlib does not allow alpha to be anything but a scalar
c_mars = numpy.asarray([(0, 0, 1, a) for a in alpha_mars])

# 3D celestial sphere of mars
theta_mars = numpy.pi / 2 - dec_mars
phi_mars = ra_mars
mars_traj = [numpy.cos(phi_mars) * numpy.sin(theta_mars), numpy.sin(phi_mars) * numpy.sin(theta_mars),
            numpy.cos(theta_mars)]

mars_ster_traj = filterForSterProjection(theta_mars,phi_mars,theta0,phi0)



# jupiter data
jup_arr = numpy.load('10_Years/jupiter4.npy')
ra_jup = jup_arr[0]
dec_jup = jup_arr[1]

T_jup = 103894  # hours
t_jup = (-1 / T_jup) * numpy.array(range(0, len(ra_jup)))
alpha_jup = numpy.array(numpy.exp(t_jup))

# way to get around the fact that matplotlib does not allow alpha to be anything but a scalar
c_jup = numpy.asarray([(0, 0, 1, a) for a in alpha_jup])

# 3D celestial sphere of jupiter
theta_jup = numpy.pi / 2 - dec_jup
phi_jup = ra_jup
jup_traj = [numpy.cos(phi_jup) * numpy.sin(theta_jup), numpy.sin(phi_jup) * numpy.sin(theta_jup), numpy.cos(theta_jup)]

jup_ster_traj = filterForSterProjection(theta_jup,phi_jup,theta0,phi0)



# saturn data
sat_arr = numpy.load('10_Years/saturn4.npy')
ra_sat = sat_arr[0]
dec_sat = sat_arr[1]

T_sat = 258216  # hours
t_sat = (-1 / T_sat) * numpy.array(range(0, len(ra_sat)))
alpha_sat = numpy.array(numpy.exp(t_sat))

# way to get around the fact that matplotlib does not allow alpha to be anything but a scalar
c_sat = numpy.asarray([(0, 0, 1, a) for a in alpha_sat])

# 3D celestial sphere of saturn
theta_sat = numpy.pi / 2 - dec_sat
phi_sat = ra_sat
sat_traj = [numpy.cos(phi_sat) * numpy.sin(theta_sat), numpy.sin(phi_sat) * numpy.sin(theta_sat), numpy.cos(theta_sat)]

sat_ster_traj = filterForSterProjection(theta_sat,phi_sat,theta0,phi0)



# uranus data
ura_arr = numpy.load('10_Years/uranus4.npy')
ra_ura = ura_arr[0]
dec_ura = ura_arr[1]

T_ura = 736440  # hours
t_ura = (-1 / T_ura) * numpy.array(range(0, len(ra_ura)))
alpha_ura = numpy.array(numpy.exp(t_ura))

# way to get around the fact that matplotlib does not allow alpha to be anything but a scalar
c_ura = numpy.asarray([(0, 0, 1, a) for a in alpha_ura])

# 3D celestial sphere of uranus
theta_ura = numpy.pi / 2 - dec_ura
phi_ura = ra_ura
ura_traj = [numpy.cos(phi_ura) * numpy.sin(theta_ura), numpy.sin(phi_ura) * numpy.sin(theta_ura), numpy.cos(theta_ura)]

ura_ster_traj = filterForSterProjection(theta_ura,phi_ura,theta0,phi0)



# neptune data
nep_arr = numpy.load('10_Years/neptune4.npy')
ra_nep = nep_arr[0]
dec_nep = nep_arr[1]

T_nep = 1440368  # hours
t_nep = (-1 / T_nep) * numpy.array(range(0, len(ra_nep)))
alpha_nep = numpy.array(numpy.exp(t_nep))

# way to get around the fact that matplotlib does not allow alpha to be anything but a scalar
c_nep = numpy.asarray([(0, 0, 1, a) for a in alpha_nep])

# 3D celestial sphere of neptune
theta_nep = numpy.pi / 2 - dec_nep
phi_nep = ra_nep
nep_traj = [numpy.cos(phi_nep) * numpy.sin(theta_nep), numpy.sin(phi_nep) * numpy.sin(theta_nep), numpy.cos(theta_nep)]

nep_ster_traj = filterForSterProjection(theta_nep,phi_nep,theta0,phi0)



# pluto data
plu_arr = numpy.load('10_Years/pluto4.npy')
ra_plu = plu_arr[0]
dec_plu = nep_arr[1]

T_plu = 2172480  # hours
t_plu = (-1 / T_plu) * numpy.array(range(0, len(ra_plu)))
alpha_plu = numpy.array(numpy.exp(t_plu))

# way to get around the fact that matplotlib does not allow alpha to be anything but a scalar
c_plu = numpy.asarray([(0, 0, 1, a) for a in alpha_plu])

# 3D celestial sphere of pluto
theta_plu = numpy.pi / 2 - dec_plu
phi_plu = ra_plu
plu_traj = [numpy.cos(phi_plu) * numpy.sin(theta_plu), numpy.sin(phi_plu) * numpy.sin(theta_plu), numpy.cos(theta_plu)]

plu_ster_traj = filterForSterProjection(theta_plu,phi_plu,theta0,phi0)









#Fig = plt.figure()
#
#Ax = Fig.add_subplot(111, projection='3d')
#
#X_1 = stars_traj[0]
#Y_1 = stars_traj[1]
#Z_1 = stars_traj[2]
#
#X_2 = merc_traj[0]
#Y_2 = merc_traj[1]
#Z_2 = merc_traj[2]
#
#X_3 = ven_traj[0]
#Y_3 = ven_traj[1]
#Z_3 = ven_traj[2]
#
#X_4 = mars_traj[0]
#Y_4 = mars_traj[1]
#Z_4 = mars_traj[2]
#
#Ax.scatter(X_1, Y_1, Z_1, c='b', marker='*', s=s)
#Ax.scatter(X_2, Y_2, Z_2, c='c', marker='o', edgecolors=c_merc)
#Ax.scatter(X_3, Y_3, Z_3, c='m', marker='o', edgecolors=c_ven)
#Ax.scatter(X_4, Y_4, Z_4, c='r', marker='o', edgecolors=c_mars)
##
#
#
# ##Stereographic projection
#
f = plt.figure()
ax = plt.subplot()

X_1 = stars_stereograph[0]*numpy.cos(stars_stereograph[1])
Y_1 = stars_stereograph[0]*numpy.sin(stars_stereograph[1])

plt.scatter(X_1, Y_1, marker='*', c='yellow', s=s)
ax.set_facecolor('black')


#ax.set_yticklabels([])


#ax = plt.gca()
#ax.set_xlabel('Declination (rad)', fontsize=12)
#ay = plt.gca()
#ay.set_ylabel('Right Ascension (rad)', fontsize=12)

X_2 = merc_ster_traj[0]*numpy.cos(merc_ster_traj[1])
Y_2 = merc_ster_traj[0]*numpy.sin(merc_ster_traj[1])

#print (X_2,Y_1)

X_3 = ven_ster_traj[0]*numpy.cos(ven_ster_traj[1])
Y_3 = ven_ster_traj[0]*numpy.sin(ven_ster_traj[1])


X_4 = mars_ster_traj[0]*numpy.cos(mars_ster_traj[1])
Y_4 = mars_ster_traj[0]*numpy.sin(mars_ster_traj[1])


#fig2 =  plt.figure()
#ax2 = fig.add_subplot(111)

#ax2.fig.add_subplot(111, projection = '2d')

#scat1 = ax.scatter(X_1, Y_1, c='b', marker='*', s=s)
scat2=ax.scatter(X_2, Y_2, c='c', alpha = 0.35, marker='o', edgecolors=c_merc, label = 'mercury')
scat3=ax.scatter(X_3, Y_3, c='m', alpha = 0.35, marker='o', edgecolors=c_ven, label = 'venus')
scat4=ax.scatter(X_4, Y_4, c='r', alpha = 0.35, marker='o', edgecolors=c_mars, label = 'mars')
plt.legend(loc = 'upper right')
plt.title('10 Years from  18:00 18-5-96')
plt.xlabel('West to East (rad)')
plt.ylabel('South to North (rad)')
plt.xlim([0, 1])
plt.ylim([0, 0.75])
opacity = 0.8
tail = 30

def init():
#   scat1.set_offsets([])
   scat2.set_offsets([])
   scat3.set_offsets([])
   scat4.set_offsets([])
   return scat2, scat3, scat4, #scat1,



def animate(i):
#   data1 = numpy.hstack((X_1[i-tail:i,numpy.newaxis], Y_1[i-tail:i, numpy.newaxis]))
   data2 = numpy.hstack((X_2[i-tail:i,numpy.newaxis], Y_2[i-tail:i, numpy.newaxis]))
   data3 = numpy.hstack((X_3[i-tail:i,numpy.newaxis], Y_3[i-tail:i, numpy.newaxis]))
   data4 = numpy.hstack((X_4[i-tail:i,numpy.newaxis], Y_4[i-tail:i, numpy.newaxis]))
#   scat1.set_offsets(data1)
   scat2.set_offsets(data2)
   scat3.set_offsets(data3)
   scat4.set_offsets(data4)
   return scat2, scat3, scat4, #scat1,

anim = animation.FuncAnimation(f, animate, init_func=init, frames=len(X_1)+1,
                              interval=90, blit=False, repeat=False)

#plt.axis('off')

#plt.show()

#fig_size = plt.rcParams["figure.figsize"]
#fig_size[0] = 12
#fig_size[1] = 11

#plt.style.use("seaborn-paper")


#animate on the disk


#
#fig = plt.figure()
#ax = plt.axes(xlim=(0, 2*numpy.pi), ylim=(-0.5, 0.5))
#scat1 = ax.scatter([], [], s=15, alpha = opacity, c = 'r', label = 'Stars')
#scat2 = ax.scatter([], [], s=15, alpha = opacity, c = 'b', label = 'Mercury')
#scat3 = ax.scatter([], [], s=15, alpha = opacity, c = 'c', label = 'Venus')
#scat4 = ax.scatter([], [], s=15, alpha = opacity, c = 'm', label = 'Mars')
#plt.legend(loc = 'upper right')
#plt.title('10 Years from  18:00 18-5-96')
#plt.xlabel('Arbitrary units West to East')
#plt.ylabel('Arbitrary units South to North')
#
#
#def init():
#   scat1.set_offsets([])
#   scat2.set_offsets([])
#   scat3.set_offsets([])
#   scat4.set_offsets([])
#   return scat1, scat2, scat3, scat4
#
#
#
#def animate(i):
#   data1 = numpy.hstack((X_1[i-tail:i,numpy.newaxis], Y_1[i-tail:i, numpy.newaxis]))
#   data2 = numpy.hstack((X_2[i-tail:i,numpy.newaxis], Y_2[i-tail:i, numpy.newaxis]))
#   data3 = numpy.hstack((X_3[i-tail:i,numpy.newaxis], Y_3[i-tail:i, numpy.newaxis]))
#   data4 = numpy.hstack((X_4[i-tail:i,numpy.newaxis], Y_4[i-tail:i, numpy.newaxis]))
#   scat1.set_offsets(data1)
#   scat2.set_offsets(data2)
#   scat3.set_offsets(data3)
#   scat4.set_offsets(data4)
#   return scat1, scat2, scat3, scat4
#
#anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(X_1)+1,
#                              interval=50, blit=False, repeat=False)
#
#
