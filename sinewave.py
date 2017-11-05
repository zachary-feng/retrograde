from astropy.coordinates import solar_system_ephemeris
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import get_body_barycentric, get_body, get_moon
import astropy.coordinates as c
import astropy.time as t
import matplotlib.animation as animation

plt.style.use("seaborn-paper")

######################## data generation segment below ########################

#solar_system_ephemeris.set('jpl') 
#
##Total time in hours
#evolve = 8760
#timeStep = 100
#numDataPoints = int(evolve/timeStep)
#
##load the Montreal's location data, we're at negative lon because we're west.
#loc = c.EarthLocation(lat=45.5017*u.deg, lon= -73.5673*u.deg, height=271*u.m) 
#
#
## Setting Eastern Daylight Time
#univTimeOffset = -4*u.hour 
#
##Setting the time, relative to UTC/GMT, I wanna start at 6PM.
#night = t.Time('1996-5-18 18:00:00') - univTimeOffset 
#
#
##Turning the night into a line space to plot
#d_time = np.arange(0, evolve, timeStep)*u.hour
#
##frame = c.AltAz(obstime=night+d_time, location=loc)
#
#evolveValue = np.array([])

#np.ndarray(shape=(2,2), dtype=float, order='F')

#mercuryArr = np.ndarray(shape = (2, numDataPoints), dtype = float)
#venusArr = np.ndarray(shape = (2, numDataPoints), dtype = float)
#marsArr = np.ndarray(shape = (2, numDataPoints), dtype = float)
#jupiterArr = np.ndarray(shape = (2, numDataPoints), dtype = float)
#saturnArr = np.ndarray(shape = (2, numDataPoints), dtype = float)
#uranusArr = np.ndarray(shape = (2, numDataPoints), dtype = float)
#neptuneArr = np.ndarray(shape = (2, numDataPoints), dtype = float)
#plutoArr = np.ndarray(shape = (2, numDataPoints), dtype = float)

#print(float(get_body('mercury', night, loc).dec.to(u.rad)/u.rad))

#load the object from astropy library
#for i in range(numDataPoints):
#    evolveValue = np.append(evolveValue,night+d_time[i])
    
#    mercuryArr[0][i] = get_body('mercury', evolveValue[i], loc).ra.to(u.rad)/u.rad
#    mercuryArr[1][i] = get_body('mercury', evolveValue[i], loc).dec.to(u.rad)/u.rad
#    
#    venusArr[0][i] = get_body('venus', evolveValue[i], loc).ra.to(u.rad)/u.rad
#    venusArr[1][i] = get_body('venus', evolveValue[i], loc).dec.to(u.rad)/u.rad
    
#    marsArr[0][i] = get_body('mars', evolveValue[i], loc).ra.to(u.rad)/u.rad
#    marsArr[1][i] = get_body('mars', evolveValue[i], loc).dec.to(u.rad)/u.rad
#    
#    jupiterArr[0][i] = get_body('jupiter', evolveValue[i], loc).ra.to(u.rad)/u.rad
#    jupiterArr[1][i] = get_body('jupiter', evolveValue[i], loc).dec.to(u.rad)/u.rad
#    
#    saturnArr[0][i] = get_body('saturn', evolveValue[i], loc).ra.to(u.rad)/u.rad
#    saturnArr[1][i] = get_body('saturn', evolveValue[i], loc).dec.to(u.rad)/u.rad
#
#    uranusArr[0][i] = get_body('uranus', evolveValue[i], loc).ra.to(u.rad)/u.rad
#    uranusArr[1][i] = get_body('uranus', evolveValue[i], loc).dec.to(u.rad)/u.rad
#    
#    neptuneArr[0][i] = get_body('neptune', evolveValue[i], loc).ra.to(u.rad)/u.rad
#    neptuneArr[1][i] = get_body('neptune', evolveValue[i], loc).dec.to(u.rad)/u.rad
#    
#    plutoArr[0][i] = get_body('pluto', evolveValue[i], loc).ra.to(u.rad)/u.rad
#    plutoArr[1][i] = get_body('pluto', evolveValue[i], loc).dec.to(u.rad)/u.rad
#


#Saving the data
#np.savetxt('mercury3.txt', (mercuryArr).reshape(-1, 2), delimiter = '\t')
#np.savetxt('venus3.txt', (venusArr).reshape(-1, 2), delimiter = '\t')
#np.savetxt('mars3.txt', (marsArr).reshape(-1, 2), delimiter = '\t')
#np.savetxt('jupiter3.txt', (jupiterArr).reshape(-1, 2), delimiter = '\t')
#np.savetxt('saturn3.txt', (saturnArr).reshape(-1, 2), delimiter = '\t')
#np.savetxt('uranus3.txt', (uranusArr).reshape(-1, 2), delimiter = '\t')
#np.savetxt('neptune3.txt', (neptuneArr).reshape(-1, 2), delimiter = '\t')
#np.savetxt('pluto3.txt', (plutoArr).reshape(-1, 2), delimiter = '\t')

#
##Plotting
##plt.plot(d_time, mercuryArr[1])
##plt.plot(d_time, mercuryArr[0])
##plt.plot(mercuryArr[0], mercuryArr[1],'r.', label= r'Mercury')
##plt.plot(venusArr[0], venusArr[1], 'b.',label= r'Venus')
#plt.plot(marsArr[0], marsArr[1], 'r-',label= r'Mars')
##plt.plot(jupiterArr[0], jupiterArr[1], 'c.', label= r'Jupiter')
#plt.legend(loc='best')
#
##plt.xlim(0, 12)
#plt.ylim(-0.5, 0.5)
#plt.title(str(evolve/24)+ ' Days from  18:00 18-5-96')
#plt.xlabel('Right Ascension (rad)')
#plt.ylabel('Declination (rad)')
#plt.show()

######################## data generation segment above ########################

from numpy import genfromtxt

#def _update_plot(i, fig, scat):
    #scat1.set_offsets(([0, i], [50, i], [100, i]))
    #scat1.set_offsets(([0, i], [50, i], [100, i]))
    #scat2.set_offsets(([0, i], [50, i], [100, i]))
    #scat3.set_offsets(([0, i], [50, i], [100, i]))
    #print('Frames %d' %i)

# load numpy files into arrays

data1 = np.load('planets4/mercury4.npy')
data2 = np.load('planets4/venus4.npy')
data3 = np.load('planets4/mars4.npy')
data4 = np.load('planets4/jupiter4.npy')
data5 = np.load('planets4/saturn4.npy')
data6 = np.load('planets4/uranus4.npy')
data7 = np.load('planets4/neptune4.npy')
data8 = np.load('planets4/pluto4.npy')
#data1 = genfromtxt('mars.txt', delimiter = '\t')
#data2 = genfromtxt('venus.txt', delimiter = '\t')
#data3 = genfromtxt('mercury.txt', delimiter = '\t')
#data4 = genfromtxt('jupiter.txt', delimiter = '\t')

fig = plt.figure()

# load arrays into x and y arrays

x1 = data1[0]
y1 = data1[1]

x2 = data2[0]
y2 = data2[1]

x3 = data3[0]
y3 = data3[1]

x4 = data4[0]
y4 = data4[1]

x5 = data5[0]
y5 = data5[1]

x6 = data6[0]
y6 = data6[1]

x7 = data7[0]
y7 = data7[1]

x8 = data8[0]
y8 = data8[1]

#x1 = data1[:,0]
#y1 = data1[:,1]

#x2 = data2[:,0]
#y2 = data2[:,1]

#x3 = data3[:,0]
#y3 = data3[:,1]

#x4 = data4[:,0]
#y4 = data4[:,1]

plt.figure(1)
plt.scatter(x1, y1, s=60)
plt.scatter(x2, y2, s=60)
plt.scatter(x3, y3, s=60)
plt.scatter(x4, y4, s=60)
plt.scatter(x5, y5, s=60)
plt.scatter(x6, y6, s=60)
plt.scatter(x7, y7, s=60)
plt.scatter(x8, y8, s=60)
#py.axis([0, 1, 0, 1])
#py.show()

# animation segment

fig = plt.figure(2)
ax = plt.axes(xlim=(0, 6), ylim=(-.75, .75))
scat1 = ax.scatter([], [], s=15, alpha = 1, c = 'lightpink', label = 'mercury')
scat2 = ax.scatter([], [], s=15, alpha = 1, c = 'orange', label = 'venus')
scat3 = ax.scatter([], [], s=15, alpha = 1, c = 'red', label = 'mars')
scat4 = ax.scatter([], [], s=15, alpha = 1, c = 'chocolate', label = 'jupiter')
scat5 = ax.scatter([], [], s=15, alpha = 1, c = 'khaki', label = 'saturn')
scat6 = ax.scatter([], [], s=15, alpha = 1, c = 'lightblue', label = 'uranus')
scat7 = ax.scatter([], [], s=15, alpha = 1, c = 'blue', label = 'neptune')
scat8 = ax.scatter([], [], s=15, alpha = 1, c = 'dimgrey', label = 'pluto')
plt.legend(loc = 'lower left')
plt.title('10 Years from  18:00 18-5-96')
plt.xlabel('Right Ascension (rad)')
plt.ylabel('Declination (rad)')

def init():
    scat1.set_offsets([])
    scat2.set_offsets([])
    scat3.set_offsets([])
    scat4.set_offsets([])
    scat5.set_offsets([])
    scat6.set_offsets([])
    scat7.set_offsets([])
    scat8.set_offsets([])
    return scat1, scat2, scat3, scat4, scat5, scat6, scat7, scat8

def animate(i):
    # hstack transposes matrix and resaves into data arrays
    data1 = np.hstack((x1[i-5:i,np.newaxis], y1[i-5:i, np.newaxis]))
    data2 = np.hstack((x2[i-5:i,np.newaxis], y2[i-5:i, np.newaxis]))
    data3 = np.hstack((x3[i-5:i,np.newaxis], y3[i-5:i, np.newaxis]))
    data4 = np.hstack((x4[i-5:i,np.newaxis], y4[i-5:i, np.newaxis]))
    data5 = np.hstack((x5[i-5:i,np.newaxis], y5[i-5:i, np.newaxis]))
    data6 = np.hstack((x6[i-5:i,np.newaxis], y6[i-5:i, np.newaxis]))
    data7 = np.hstack((x7[i-5:i,np.newaxis], y7[i-5:i, np.newaxis]))
    data8 = np.hstack((x8[i-5:i,np.newaxis], y8[i-5:i, np.newaxis]))
    # set_offsets plots the data arrays
    scat1.set_offsets(data1)
    scat2.set_offsets(data2)
    scat3.set_offsets(data3)
    scat4.set_offsets(data4)
    scat5.set_offsets(data5)
    scat6.set_offsets(data6)
    scat7.set_offsets(data7)
    scat8.set_offsets(data8)
    return scat1, scat2, scat3, scat4, scat5, scat6, scat7, scat8

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(x1)+1, 
                               interval=50, blit=False, repeat=False)
#anim.save('AllTail5_HD.mp4', writer="ffmpeg", bitrate = 1000000)

plt.show()
