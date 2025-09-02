import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.widgets import Slider

import matplotlib.pyplot as plt

# Configurações iniciais
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
ax.set_aspect('equal')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)

# Animação
dt = 20 # ms

# Orbita
radius = 1

# Ponto
angle = 0
point = Circle((0, 0), radius=0.1, color="red")
ax.add_patch(point)

# Slider para controlar a velocidade
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
speed_slider = Slider(ax_slider, 'Velocidade', 0.1, 5.0, valinit=1.0)

# Função de animação
def animate(frame):
    global angle

    speed = speed_slider.val
    
    angle += speed * 0.05 

    x = radius * np.cos(angle)
    y = radius * np.sin(angle)

    point.set_center((x, y))

    return point,

ani = FuncAnimation(fig, animate, interval=dt, blit=True, cache_frame_data=False)
plt.show()