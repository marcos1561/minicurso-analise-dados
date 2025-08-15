import numpy  as np
from particles import *
from scipy.stats import linregress

def create_sim(n, dt=0.01, d=1):
    state = BrownianState(
        pos=np.zeros((n, 2), dtype=np.float64),
        diffusion=np.array([d, d]),
    )

    sim = Simulation(state, NoForce(), InfSpace(), dt)
    return sim


def run_sim(sim: Simulation, tf):
    while sim.time < tf:
        sim.step()

def check_different_dt():
    n = 100
    tf = 40

    sims = [
        create_sim(n, 0.01),
        create_sim(n, 0.001),
    ]

    for i, s in enumerate(sims):
        s.run(tf)
        
        view = Visualizer(AnimCfg(), s)
        view.fig.savefig(f"sim_{i}.png")

def compute_msd(trajectories: np.ndarray, dt):
    """
    Calcula o Mean Squared Displacement (MSD) dado um conjunto de trajetórias.

    Parameters:
    -----------
    trajectories:
        Array de shape (n_particles, n_steps, 2), onde:
        - n_particles é o número de partículas.
        - n_steps é o número de passos temporais.
        - 2 representa as coordenadas (x, y).

    dt:
        Tempo entre dois pontos na trajetória.

    Returns:
    -------
    msd: np.ndarray
        Array de shape (n_steps,) contendo o MSD para cada intervalo de tempo.

    dt_array: np.ndarray
        Array de shape (n_steps,) contendo o intervalo de tempo para cada MSD.
    """
    # Número de passos temporais
    n_steps = trajectories.shape[1]

    # Inicializa o MSD
    msd = np.empty(n_steps - 1, dtype=np.float64)
    dt_array = np.empty(n_steps - 1, dtype=np.float64)

    # Calcula o deslocamento ao quadrado para cada intervalo de tempo
    for num_dt in range(1, n_steps):
        displacements = trajectories[:, num_dt:, :] - trajectories[:, :n_steps-num_dt, :]
        squared_displacements = np.sum(displacements**2, axis=-1)
        msd[num_dt-1] = np.mean(squared_displacements)
        dt_array[num_dt-1] = num_dt * dt

    return msd, dt_array

def msd_experiment():
    dt = 1
    tf = 200
    n = 100
    diffusion_range = np.linspace(0.3, 4, 10)

    collect_dt = 1

    num_steps = round(tf/dt)
    num_steps_to_collect = round(collect_dt/dt)
    coef_result = []
    for d in diffusion_range:
        sim = create_sim(n, dt, d)
        
        num_points = (num_steps-1) // num_steps_to_collect + 1
        trajectories = np.empty((n, num_points, 2), dtype=np.float64)
        times = np.empty(num_points, dtype=np.float64)

        count = 0
        for i in range(num_steps):
            sim.step()
            if i % num_steps_to_collect == 0:
                trajectories[:, count, :] = sim.state.pos
                times[count] = sim.time
                count += 1

        msd, dt_array = compute_msd(trajectories, times[1] - times[0])

        result = linregress(dt_array, msd)
        coef_result.append(result.slope)

    # Valore esperado
    x = np.linspace(diffusion_range[0], diffusion_range[-1], 100)
    y = 2 * x**2
    plt.plot(x, y, c="red")

    plt.scatter(diffusion_range, coef_result)
    plt.show()

# check_different_dt()
msd_experiment()