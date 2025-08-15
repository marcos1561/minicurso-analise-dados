import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.animation as animation

import numpy as np

class State:
    def __init__(self, pos: np.ndarray, vel: np.ndarray, mass: np.ndarray):
        self.pos = pos
        self.vel = vel
        self.mass = mass

    @property
    def num_particles(self):
        return self.pos.shape[0]

class BrownianState:
    def __init__(self, pos: np.ndarray, diffusion: np.ndarray):
        self.pos = pos
        self.diffusion = diffusion

    @property
    def num_particles(self):
        return self.pos.shape[0]

class Rectangle:
    def __init__(self, height, length, bottom_left):
        self.height = height
        self.length = length
        self.bottom_left = bottom_left
        self.center = bottom_left + np.array([length, height]) / 2

    def rigid_borders(self, state: State):
        rel_pos = state.pos - self.center

        x, y = rel_pos[:, 0], rel_pos[:, 1]
        out_x = np.abs(x) > self.length/2
        out_y = np.abs(y) > self.height/2

        state.vel[out_x, 0] *= -1
        state.vel[out_y, 1] *= -1

    def draw(self, ax: Axes):
        bl = self.bottom_left
        h, l = self.height, self.length

        ax.plot([bl[0], bl[0]], [bl[1], bl[1]+h], c="black")
        ax.plot([bl[0]+l, bl[0]+l], [bl[1], bl[1]+h], c="black")
        ax.plot([bl[0], bl[0]+l], [bl[1], bl[1]], c="black")
        ax.plot([bl[0], bl[0]+l], [bl[1]+h, bl[1]+h], c="black")

    def get_positions(self, nx, ny, pad=0):
        bl = self.bottom_left

        xy = np.meshgrid(
            np.linspace(bl[0] + pad, bl[0] + self.length - pad, nx),
            np.linspace(bl[1] + pad, bl[1] + self.height - pad, ny),
        )

        return np.stack(xy, axis=-1).reshape(-1, 2)
    
class InfSpace:
    def draw(self, ax):
        pass

class RigidCore:
    def __init__(self, force_mod, particle_radius):
        self.force_mod = force_mod
        self.particle_radius = particle_radius

        self.particle_d_2 = self.particle_radius**2

    def get_particle_radius(self):
        return self.particle_radius

    def calc_forces(self, state: State, forces: np.ndarray):
        particle_d_2 = self.particle_d_2
        force_mod = self.force_mod

        pos = state.pos

        for i in range(state.num_particles):
            for j in range((i+1), state.num_particles):
                dr = pos[i] - pos[j]
                dist_2 = (dr**2).sum()

                if dist_2 > particle_d_2:
                    continue

                dist = dist_2**.5

                force_ij = dr / dist * force_mod

                forces[i] = force_ij
                forces[j] = -force_ij

class NoForce:
    def calc_forces(self, state, forces):
        return

class Simulation:
    def __init__(self, state, potential_cfg: RigidCore, space_cfg: Rectangle, dt):
        self.state = state
        self.space_cfg = space_cfg
        self.potencial_cfg = potential_cfg
        self.dt = dt

        self.time = 0
        self.time_step = 0
        self.forces = np.zeros_like(self.state.pos)

        step_dict = {
            State: self.newton_step,
            BrownianState: self.brownin_step,
        }
        self._step = step_dict[type(state)]

        self.calc_forces()

    @property
    def num_particles(self):
        return self.state.num_particles    

    def calc_forces(self):
        self.forces[:] = 0        
        self.potencial_cfg.calc_forces(self.state, self.forces)

    def newton_second_law(self):
        state: State = self.state
        dt = self.dt

        pos, vel = state.pos, self.state.vel
        old_force = np.copy(self.forces)

        pos += vel * dt  + self.forces / state.mass[:, None] * dt**2/2

        self.calc_forces()

        vel += dt * (self.forces + old_force) / state.mass[:, None] /  2

    def brownian_motion(self):
        state: BrownianState = self.state

        D = state.diffusion
        gaussian_sample = np.random.randn(self.state.num_particles, 2) * D
        state.pos += gaussian_sample * self.dt**.5


    def equation_of_motion_sp(self, t, y: np.ndarray):
        particle_d_2 = 1**2
        force_mod = 1

        pos = y[:self.num_particles*2].reshape(-1, 2)
        vel = y[pos.size:].reshape(-1, 2)
        mass = self.state.mass
        accel = np.zeros_like(pos)
        
        self.space_cfg.rigid_borders(pos, vel)

        n = pos.shape[0]
        for i in range(1, n):
            for j in range((i+1), n):
                dr = pos[i] - pos[j]
                dist_2 = (dr**2).sum()

                if dist_2 > particle_d_2:
                    continue

                dist = dist_2**.5

                force_ij = dr / dist * force_mod

                accel[i] = force_ij / mass[i]
                accel[j] = -force_ij / mass[j]

        return np.concatenate([
            vel.flatten(),
            accel.flatten(),
        ])

    def newton_step(self):
        "Avança o sistema em um passo temporal"
        self.newton_second_law()
        self.space_cfg.rigid_borders(self.state)
        self.calc_forces()

    def brownin_step(self):
        self.brownian_motion()

    def step(self):
        self._step()
        self.time += self.dt
        self.time_step += 1

    def run(self, tf):
        while self.time < tf:
            self.step()


class AnimCfg:
    def __init__(self, num_steps_per_frame=10):
        self.num_steps_per_frame = num_steps_per_frame

class Visualizer:
    def __init__(self, cfg: AnimCfg, sim: Simulation):
        self.sim = sim
        self.cfg = cfg

        self.fig, self.ax = plt.subplots()

        pos = self.sim.state.pos
        self.points = self.ax.scatter(pos[:, 0], pos[:, 1], c=np.random.rand(sim.num_particles))
        self.sim.space_cfg.draw(self.ax)

    def update(self, frame):
        for _ in range(self.cfg.num_steps_per_frame):
            self.sim.step()
        
        self.points.set_offsets(self.sim.state.pos)

    def animate(self):
        anim = animation.FuncAnimation(self.fig, self.update, interval=1/60*1000, cache_frame_data=False)
        plt.show()


if __name__ == "__main__":
    # nx = 5
    # ny = 5
    # n = nx * ny

    # xy = np.meshgrid(
    #     np.arange(nx),
    #     np.arange(ny),
    # )

    # pos = np.stack(xy, axis=-1).reshape(-1, 2).astype(np.float64)

    # angles = np.random.rand(n) * 2 * np.pi
    # vel = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    # mass = np.full(n, 1.0)
    

    # state = State(pos, vel, mass)

    space_cfg = Rectangle(
        height=10, length=10,
        bottom_left=np.zeros(2),
    )

    potential_cfg = RigidCore(
        force_mod=10,
        particle_radius=0.5,
    )

    pos = space_cfg.get_positions(5, 5, pad=potential_cfg.get_particle_radius())
    
    n = pos.shape[0]
    angles = np.random.rand(n) * 2 * np.pi
    vel = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    mass = np.full(n, 1.0)
    
    state = State(pos, vel, mass)

    dt = 0.01

    sim = Simulation(state, potential_cfg, space_cfg, dt)

    view = Visualizer(
        cfg=AnimCfg(
            num_steps_per_frame=50,
        ),
        sim=sim,
    )

    view.animate()