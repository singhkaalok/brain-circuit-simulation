import numpy as np
import matplotlib.pyplot as plt

#Initialize all variables
a = 0.02
b = 0.2
c = -50
d = 2

v = -65
u = v*b

simulation_time = 1000
memvolt = np.zeros(simulation_time)
Iall = np.zeros(simulation_time)


#for loop over simulation time
for t in range(simulation_time):
#Define the input strength 
  I = -2 if (t>200) & (t<400) else 7

#Check if there is an action potential
  if v >= 30:
    v = c
    u += d

#update the variable membrane
  v += 0.04*v**2 + 5*v + 140 - u + I
  u += a*(b*v - u)

#Collect the variables for subsequent plottings
  memvolt[t] = v
  Iall[t] = I

#Plotting
fig,ax = plt.subplots(1, figsize=(15,7))
plt.plot(memvolt,'k', label='Membrane Potential')
plt.plot(Iall-100,'m', label='Stimulation')
plt.box(False)
plt.xlim([0,simulation_time])
plt.legend(fontsize = 15)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.show()

#Excitatory Cells                                   #Inhibitory cells
Ne = 800;                                            Ni = 200
re = np.random.rand(Ne)**2;                          ri = np.random.rand(Ni)
a = np.hstack((0.02*np.ones(Ne),                     0.02+0.08*ri))
b = np.hstack((0.2*np.ones(Ne),                      0.25-0.05*ri))
c = np.hstack((-65+15*re,                            -65*np.ones(Ni)))
d = np.hstack((8-6*re,                               2*np.ones(Ni)))

v = -65*np.ones(Ne+Ni)
u = b*v

#s matrix connectivity
S = np.hstack((0.5*np.random.rand(Ne+Ni,Ne), -np.random.rand(Ne+Ni,Ne)))

simulation_time = 5000
firings = np.array([[],[]])
for t in range (simulation_time):

  #define the exogenous input
  I = np.hstack((5*np.random.randn(Ne),2*np.random.randn(Ni)))

  #check for action potentials
  fired = np.where(v>=30)[0]

  #store the spikes indices and times
  tmp = np.stack((np.tile(t, len(fired)),fired))
  firings = np.concatenate((firings,tmp),axis=1) 
  #update membrane variables for neurons that spiked
  v[fired] = c[fired]
  u[fired] += d[fired]

  #update the I to include the spiking activity
  I = I + np.sum(S[:,fired], axis=1)

  #update membrane neurons for all neurons
  v += 0.04*v**2 + 5*v + 140 - u + I
  u += a*(b*v - u)
  
fig = plt.subplots(1, figsize=(10,6))
plt.plot(firings[0,:], firings[1,:], 'k.', markersize=.5)
plt.show()

popact = np.zeros(simulation_time)

#get the population response
for t in range(simulation_time):
  popact[t] = 100*np.sum(firings[0,:]==t) / (Ne+Ni)

#spectral analysis
popactX = np.abs(np.fft.fft(popact-np.mean(popact)))**2
hz = np.linspace(0,1000/2,int(simulation_time/2+1))

#ploting  
fig,ax = plt.subplots(1,2, figsize=(15,5))
ax[0].plot(popact)
ax[0].set_xlabel('Time (ms)')
ax[0].set_ylabel('Percentage of  Neuron Active')
ax[0].set_title('Time Domain')
ax[1].plot(hz,popactX[:len(hz)])
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Spectral Power')
ax[1].set_title('Frequency Domain')
ax[1].set_xlim([0,60])
plt.show()

#call a python function
def simCircuit(I):
  firings = np.array([[],[]])
  #Reinitliaze membrane variables
  v = -65*np.ones(Ne+Ni)
  u = b*v


  for t in range (len(I)):

    #define the exogenous input
    stim = np.hstack((4*np.random.randn(Ne),1*np.random.randn(Ni)))
    stim = stim + I[t]
    #check for action potentials
    fired = np.where(v>=30)[0]

    #store the spikes indices and times
    tmp = np.stack((np.tile(t, len(fired)),fired))
    firings = np.concatenate((firings,tmp),axis=1) 
    #update membrane variables for neurons that spiked
    v[fired] = c[fired]
    u[fired] += d[fired]

    #update the I to include the spiking activity
    stim += np.sum(S[:,fired], axis=1)

    #update membrane neurons for all neurons
    v += 0.04*v**2 + 5*v + 140 - u + stim
    u += a*(b*v - u)
  return firings
  
def plotPopActivity(firings):
  npnts = int(np.max(firings[0,:])+1)

#get the population response
  popact = np.zeros(npnts)
  for t in range(npnts):
    popact[t] = 100*np.sum(firings[0,:]==t) / (Ne+Ni)

#spectral analysis
  popactX = np.abs(np.fft.fft(popact-np.mean(popact)))**2
  hz = np.linspace(0,1000/2,int(npnts/2+1))
    
  #ploting  

  fig,ax = plt.subplots(1,3, figsize=(15,5))
  ax[0].plot(firings[0,:], firings[1,:], 'k.', markersize=.5)
  ax[0].plot(I*50+100,'m',linewidth=2)
  ax[0].set_title('All neurons Fires')
  ax[1].plot(popact)
  ax[1].set_xlabel('Time (ms)')
  ax[1].set_ylabel('Percentage of  Neuron Active')
  ax[1].set_title('Time Domain')
  ax[2].plot(hz,popactX[:len(hz)])
  ax[2].set_xlabel('Frequency (Hz)')
  ax[2].set_ylabel('Spectral Power')
  ax[2].set_title('Frequency Domain')
  ax[2].set_xlim([0,60])
  plt.show()
  
  #experiment 1
I = np.ones(1200)
I[400:601] = -2

#run the simulation and visualize the results
networkSpikes = simCircuit(I)
plotPopActivity(networkSpikes)

