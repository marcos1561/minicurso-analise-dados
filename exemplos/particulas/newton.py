from particles import *

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
        num_steps_per_frame=10,
    ),
    sim=sim,
)

view.animate()