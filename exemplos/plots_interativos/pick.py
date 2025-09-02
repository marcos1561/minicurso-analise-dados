import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot(
    np.random.rand(100), 'o',
    picker=True, pickradius=5
) 

def onpick(event):
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    points = tuple(zip(xdata[ind], ydata[ind]))
    print('Pontos Selecionados:', points)

fig.canvas.mpl_connect('pick_event', onpick)

plt.show()