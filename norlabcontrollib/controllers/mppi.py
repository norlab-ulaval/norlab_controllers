from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from norlabcontrollib.controllers.controller import Controller
from norlabcontrollib.models.ideal_diff_drive import IdealDiffDrive
from norlabcontrollib.util.util_func import interp_angles, wrap2pi


class MPPI(Controller):
    def __init__(self, parameter_map):
        super().__init__(parameter_map)

        self.horizon_length = parameter_map['horizon_length']
        self.n_samples = parameter_map['n_samples']
        self.n_iterations = parameter_map['n_iterations']
        self.temperature = parameter_map['temperature']
        self.damping = parameter_map['damping']
        self.control_sample_std_ratio = parameter_map['control_sample_std_ratio']
        self.id_window_size = parameter_map['id_window_size']
        self.state_cost_translational = parameter_map['state_cost_translational']
        self.state_cost_rotational = parameter_map['state_cost_rotational']
        self.input_cost_wheel = parameter_map['input_cost_wheel']
        self.wheel_radius = parameter_map['wheel_radius']
        self.baseline = parameter_map['baseline']
        self.random_seed = parameter_map.get('random_seed', 0)

        self.n_inputs = 2
        self.dt = 1.0 / self.rate

        self.motion_model = IdealDiffDrive(self.wheel_radius, self.baseline, self.dt)
        self.jacobian_3x2 = jnp.array(self.motion_model.jacobian_3x2)
        self.max_wheel_vel = float(
            self.motion_model.compute_wheel_vels(np.array([self.maximum_linear_velocity, 0]))[0]
        )
        # control_sample_std_ratio is a fraction of max_wheel_vel, since both wheels share the same limit
        self.control_sample_std = self.control_sample_std_ratio * self.max_wheel_vel

        self.accum_matrix = jnp.triu(jnp.ones((self.horizon_length, self.horizon_length)))
        self.rng_key = jax.random.PRNGKey(self.random_seed)
        self.a_opt = jnp.zeros((self.horizon_length, self.n_inputs))

        self.next_path_idx = 0
        self.linear_distance_to_goal = float('inf')
        self.euclidean_distance_to_goal = float('inf')
        self.angular_distance_to_goal = np.pi

    def new_key(self):
        self.rng_key, subkey = jax.random.split(self.rng_key)
        return subkey

    def update_path(self, new_path):
        super().update_path(new_path)
        self.next_path_idx = 0
        self.linear_distance_to_goal = float('inf')
        self.euclidean_distance_to_goal = float('inf')
        self.angular_distance_to_goal = np.pi

    def compute_desired_trajectory(self, state):
        closest_pose, self.next_path_idx = self.path.compute_orthogonal_projection(
            state, self.next_path_idx, self.id_window_size,
            self.maximum_linear_velocity, self.maximum_angular_velocity
        )
        self.closest_pose = closest_pose

        horizon_duration = self.horizon_length / self.rate
        horizon_poses, cumul_duration = self.path.compute_horizon(
            closest_pose, self.next_path_idx, horizon_duration,
            self.maximum_linear_velocity, self.maximum_angular_velocity
        )
        # TODO: Experimental change, remove the clipping of horizon_duration so it stops slowing too aggressively when reaching end of path
        horizon_duration = min(horizon_duration, cumul_duration[-1])
        interp_duration = np.linspace(0, horizon_duration, self.horizon_length)
        interp_x = np.interp(interp_duration, cumul_duration, horizon_poses[:, 0])
        interp_y = np.interp(interp_duration, cumul_duration, horizon_poses[:, 1])
        interp_yaw = interp_angles(interp_duration, cumul_duration, horizon_poses[:, 2])
        self.target_trajectory = np.stack([interp_x, interp_y, interp_yaw], axis=1)

    def compute_distance_to_goal(self, state, orthogonal_projection_id):
        self.euclidean_distance_to_goal = np.linalg.norm(self.path.poses[-1, :2] - state[:2])
        distance_to_goal_path = self.path.distances_to_goal[orthogonal_projection_id]
        distance_to_next_node = np.linalg.norm(self.path.poses[orthogonal_projection_id, :2] - state[:2])
        self.linear_distance_to_goal = distance_to_goal_path + distance_to_next_node
        self.angular_distance_to_goal = np.abs(wrap2pi(self.path.poses[-1, 5] - state[5]))

    def goal_reached(self):
        return (self.linear_distance_to_goal < self.linear_goal_tolerance) and \
               (np.abs(self.angular_distance_to_goal) < self.angular_goal_tolerance)

    @partial(jax.jit, static_argnums=(0,))
    def dynamics(self, state, action):
        # state: [x, y, yaw], action: [omega_left, omega_right] (wheel angular velocities)
        action = jnp.clip(action, -self.max_wheel_vel, self.max_wheel_vel)
        body_vel = self.jacobian_3x2 @ action  # [v_x, v_y(=0), omega] in body frame
        yaw = state[2]
        cos_yaw, sin_yaw = jnp.cos(yaw), jnp.sin(yaw)
        world_vel = jnp.array([
            body_vel[0] * cos_yaw - body_vel[1] * sin_yaw,
            body_vel[0] * sin_yaw + body_vel[1] * cos_yaw,
            body_vel[2],
        ])
        return state + world_vel * self.dt

    @partial(jax.jit, static_argnums=(0,))
    def rollout(self, state, actions):
        # actions: [horizon_length, n_inputs] -> states: [horizon_length, 3]
        states = []
        for t in range(self.horizon_length):
            state = self.dynamics(state, actions[t, :])
            states.append(state)
        return jnp.asarray(states)

    @partial(jax.jit, static_argnums=(0,))
    def reward_fn(self, states, reference_traj, actions, previous_action):
        # states, reference_traj: [horizon_length, 3] ([x, y, yaw])
        # actions: [horizon_length, n_inputs], previous_action: [n_inputs] (last action actually sent to the robot)
        translational_error = jnp.linalg.norm(states[:, :2] - reference_traj[:, :2], axis=1)
        rotational_error = jnp.abs(wrap2pi(states[:, 2] - reference_traj[:, 2]))

        action_prev = jnp.concatenate([previous_action[None, :], actions[:-1, :]], axis=0)
        input_change_cost = jnp.sum((actions - action_prev) ** 2, axis=1)

        return -(self.state_cost_translational * translational_error
                  + self.state_cost_rotational * rotational_error
                  + self.input_cost_wheel * input_change_cost)

    @partial(jax.jit, static_argnums=(0,))
    def returns(self, r):
        # r: [horizon_length], reward-to-go
        return self.accum_matrix @ r

    @partial(jax.jit, static_argnums=(0,))
    def weights(self, R):
        # R: [n_samples]
        R_stdzd = (R - jnp.max(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)
        w = jnp.exp(R_stdzd / self.temperature)
        return w / jnp.sum(w)

    @partial(jax.jit, static_argnums=(0,))
    def iteration_step(self, a_opt, state, reference_traj, previous_action, rng_key):
        z = jax.random.truncated_normal(
            rng_key, -3.0, 3.0, # restrict to 3*std
            shape=(self.n_samples, self.horizon_length, self.n_inputs)
        )
        da = z * self.control_sample_std

        actions = jnp.clip(a_opt[None, :, :] + da, -self.max_wheel_vel, self.max_wheel_vel)
        states = jax.vmap(self.rollout, in_axes=(None, 0))(state, actions)  # [n_samples, horizon_length, 3]

        reward = jax.vmap(self.reward_fn, in_axes=(0, None, 0, None))(
            states, reference_traj, actions, previous_action
        )  # [n_samples, horizon_length]
        R = jax.vmap(self.returns)(reward)  # [n_samples, horizon_length]
        w = jax.vmap(self.weights, in_axes=1, out_axes=1)(R)  # [n_samples, horizon_length]

        da_opt = jax.vmap(jnp.average, in_axes=(1, None, 1))(da, 0, w)  # [horizon_length, n_inputs]
        a_opt = jnp.clip(a_opt + da_opt, -self.max_wheel_vel, self.max_wheel_vel)

        return a_opt, states

    def update(self, state, reference_traj):
        previous_action = self.a_opt[0]
        a_opt = jnp.concatenate([self.a_opt[1:, :], jnp.zeros((1, self.n_inputs))], axis=0)
        for _ in range(self.n_iterations):
            a_opt, states = self.iteration_step(a_opt, state, reference_traj, previous_action, self.new_key())

        self.a_opt = a_opt
        self.sampled_states = states
        self.optim_trajectory = self.rollout(state, a_opt)

    def compute_command_vector(self, state):
        self.planar_state = np.array([state[0], state[1], state[5]])
        self.compute_desired_trajectory(self.planar_state)

        self.update(jnp.asarray(self.planar_state), jnp.asarray(self.target_trajectory))

        wheel_input_array = np.asarray(self.a_opt[0]).reshape(2, 1)
        body_input_array = self.motion_model.compute_body_vel(wheel_input_array).astype('float64')

        self.compute_distance_to_goal(state, self.next_path_idx)
        return body_input_array.reshape(2)


if __name__ == "__main__":

    import time
    from norlabcontrollib.path.path import Path

    parameter_map = {
        'rate': 20.0,  # control loop frequency [Hz], > 0
        'minimum_linear_velocity': 0.1,  # required by the base Controller, unused by MPPI itself [m/s], >= 0
        'maximum_linear_velocity': 1.0,  # used to derive max_wheel_vel and the reference horizon speed [m/s], > 0
        'maximum_angular_velocity': 1.0,  # used to size the reference horizon's travel time [rad/s], > 0
        'linear_goal_tolerance': 0.1,  # distance under which the goal is considered reached [m], > 0
        'angular_goal_tolerance': 0.1,  # heading error under which the goal is considered reached [rad], > 0
        'horizon_length': 40,  # number of timesteps planned/rolled out per iteration, integer > 0
        'n_samples': 4000,  # number of randomly perturbed trajectories sampled per iteration, integer > 0
        'n_iterations': 3,  # number of resample/reweight passes per control step, integer >= 1
        'temperature': 0.05,  # softmax sharpness for reward weighting, > 0 (lower = greedier on best samples)
        'damping': 0.001,  # floor added to the reward-spread normalization to avoid divide-by-zero, > 0 (small)
        'control_sample_std_ratio': 0.4, # [0,1]. 0.5 = noise std is half the max wheel speed
        'id_window_size': 50,  # search window (path indices) for the orthogonal projection lookup, integer > 0
        'state_cost_translational': 1.0,  # reward weight on position tracking error, >= 0
        'state_cost_rotational': 0.1,  # reward weight on heading tracking error, >= 0
        'input_cost_wheel': 0.0,  # reward weight penalizing wheel velocity changes between steps, >= 0
        'wheel_radius': 0.3,  # robot wheel radius [m], > 0
        'baseline': 1.2,  # distance between the left and right wheels [m], > 0
    }

    # L-shaped path with a sharp 90 degree turn, to see how the controller behaves in a corner
    segment_length = 3.0
    poses_per_meter = 40
    n_poses_per_segment = int(segment_length * poses_per_meter)

    segment_east = np.zeros((n_poses_per_segment, 6))
    segment_east[:, 0] = np.linspace(0, segment_length, n_poses_per_segment)
    segment_east[:, 5] = 0.0

    segment_north = np.zeros((n_poses_per_segment, 6))
    segment_north[:, 0] = segment_length
    segment_north[:, 1] = np.linspace(0, segment_length, n_poses_per_segment)
    segment_north[:, 5] = np.pi / 2

    path_poses = np.concatenate([segment_east, segment_north[1:]], axis=0)  # drop duplicated corner pose

    controller = MPPI(parameter_map)
    controller.update_path(Path(path_poses))

    state = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.3])
    dt = 1 / parameter_map['rate']
    max_steps = 600
    executed_trajectory = [state[:2].copy()]
    commanded_wheel_velocities = []

    import matplotlib.pyplot as plt
    plt.ion()
    fig, (ax_traj, ax_wheel) = plt.subplots(1, 2, figsize=(12, 5))

    sampled_trajectories_plot_fraction = 0.1  # plotting every sampled trajectory is slow, so subsample

    def draw_plot(step):
        ax_traj.clear()
        sampled_states = np.asarray(controller.sampled_states)  # [n_samples, horizon_length, 3]
        n_to_plot = max(1, int(sampled_states.shape[0] * sampled_trajectories_plot_fraction))
        plot_idx = np.random.choice(sampled_states.shape[0], n_to_plot, replace=False)
        # separate trajectories with a row of NaNs so they can all be drawn with a single plot call
        sampled_subset = sampled_states[plot_idx, :, :2]
        sampled_subset = np.concatenate([sampled_subset, np.full((n_to_plot, 1, 2), np.nan)], axis=1)
        sampled_subset = sampled_subset.reshape(-1, 2)
        ax_traj.plot(sampled_subset[:, 0], sampled_subset[:, 1], color='gray', alpha=0.2, linewidth=0.5)
        optim_trajectory = np.asarray(controller.optim_trajectory)  # [horizon_length, 3]
        ax_traj.plot(optim_trajectory[:, 0], optim_trajectory[:, 1], 'g-', linewidth=2, label='Optimal Trajectory')
        ax_traj.plot(path_poses[:, 0], path_poses[:, 1], 'r--', label='Reference Path')
        target_trajectory = np.asarray(controller.target_trajectory)  # [horizon_length, 3]
        ax_traj.scatter(target_trajectory[:, 0], target_trajectory[:, 1], c='orange', s=15, zorder=4,
                         label='Reference Trajectory (this iteration)')
        executed = np.array(executed_trajectory)
        ax_traj.plot(executed[:, 0], executed[:, 1], 'b-', label='Executed Trajectory')
        ax_traj.scatter(state[0], state[1], c='k', zorder=5, label='Robot')
        ax_traj.quiver(state[0], state[1], np.cos(state[5]), np.sin(state[5]), color='k', zorder=5)
        ax_traj.axis('equal')
        ax_traj.legend(loc='upper left')
        ax_traj.set_title(f"step {step}, lin_dist_to_goal={controller.linear_distance_to_goal:.3f}")

        ax_wheel.clear()
        wheel_velocities = np.array(commanded_wheel_velocities)  # [step, 2]
        if wheel_velocities.shape[0] > 0:
            steps_axis = np.arange(wheel_velocities.shape[0])
            ax_wheel.plot(steps_axis, wheel_velocities[:, 0], label='Left wheel')
            ax_wheel.plot(steps_axis, wheel_velocities[:, 1], label='Right wheel')
        ax_wheel.axhline(controller.max_wheel_vel, color='k', linestyle='--', linewidth=0.75)
        ax_wheel.axhline(-controller.max_wheel_vel, color='k', linestyle='--', linewidth=0.75)
        ax_wheel.set_xlabel('step')
        ax_wheel.set_ylabel('wheel angular velocity [rad/s]')
        ax_wheel.set_title('Commanded Wheel Velocities')
        ax_wheel.legend(loc='upper left')

        fig.canvas.draw()
        plt.pause(0.001)

    step = 0
    step_durations = []
    while not controller.goal_reached() and step < max_steps:
        step_start_time = time.perf_counter()
        cmd = controller.compute_command_vector(state)
        step_duration = time.perf_counter() - step_start_time
        step_durations.append(step_duration)
        commanded_wheel_velocities.append(np.asarray(controller.a_opt[0]))

        draw_plot(step)
        print(f"step {step}: state={state[:2]}, yaw={state[5]:.3f}, "
              f"lin_dist_to_goal={controller.linear_distance_to_goal:.3f}, "
              f"compute_time={step_duration * 1000:.1f} ms"
              + (" (includes JIT compilation)" if step == 0 else ""))
        input("Press Enter to run the next iteration...")

        v, w = cmd
        x, y, yaw = state[0], state[1], state[5]
        state = state.copy()
        state[0] = x + v * np.cos(yaw) * dt
        state[1] = y + v * np.sin(yaw) * dt
        state[5] = yaw + w * dt
        executed_trajectory.append(state[:2].copy())
        step += 1

    print(f"final state: {state}, goal_reached={controller.goal_reached()}")

    steady_state_durations = np.array(step_durations[1:]) if len(step_durations) > 1 else np.array(step_durations)
    print(f"compute_command_vector timing over {len(step_durations)} steps "
          f"(excluding first call, which includes JIT compilation):")
    print(f"  mean: {steady_state_durations.mean() * 1000:.1f} ms ({1 / steady_state_durations.mean():.1f} Hz)")
    print(f"  std:  {steady_state_durations.std() * 1000:.1f} ms")
    print(f"  min:  {steady_state_durations.min() * 1000:.1f} ms")
    print(f"  max:  {steady_state_durations.max() * 1000:.1f} ms")

    plt.ioff()
    draw_plot(step)
    plt.show()
