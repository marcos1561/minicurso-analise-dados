import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Button, Slider

# Função a ser plotada
def f(t, amplitude, frequency):
    return amplitude * np.sin(2 * np.pi * frequency * t)

t = np.linspace(0, 1, 1000)

# Parâmetros iniciais
init_amplitude = 5
init_frequency = 3

# Criando figura
fig, ax = plt.subplots()
line, = ax.plot(t, f(t, init_amplitude, init_frequency), lw=2)
ax.set_xlabel('Tempo [s]')

# Ajusta o figura para criar espaço para os sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

# Slider da frequência.
axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label='Frequência [Hz]',
    valmin=0.1,
    valmax=30,
    valinit=init_frequency,
)

# Slider da amplitude
axamp = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
amp_slider = Slider(
    ax=axamp,
    label="Amplitude",
    valmin=0,
    valmax=10,
    valinit=init_amplitude,
    orientation="vertical"
)

# Função a ser chamada quando os sliders mudam
def update(val):
    line.set_ydata(f(t, amp_slider.val, freq_slider.val))
    fig.canvas.draw_idle()


# Registrando função aos sliders
freq_slider.on_changed(update)
amp_slider.on_changed(update)

# Botão para resetar os sliders
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Resetar', hovercolor='0.975')

def reset(event):
    freq_slider.reset()
    amp_slider.reset()
button.on_clicked(reset)

plt.show()