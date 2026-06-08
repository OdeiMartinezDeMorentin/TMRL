import threading
import queue
from typing import Optional

from tminterface.client import Client
from tminterface.interface import TMInterface

import gymnasium
import numpy as np

from tmml.core.actions import Action, ACTIONS, NUM_ACTIONS, action_to_tminterface
from tmml.core.state import StateExtractor
from tmml.core.waypoints import WaypointManager
from tmml.core.rewards import RewardCalculator, RewardConfig
from tmml.tmi.client import CarState


class EnvClient(Client):
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.state = CarState()
        self._registered = False
        self._step_count = 0

        self.action_queue = queue.Queue(maxsize=1)
        self.state_queue = queue.Queue(maxsize=1)
        self.step_event = threading.Event()

        self.initial_state = None
        self._iface = None

    def on_registered(self, iface: TMInterface):
        self._iface = iface
        self._registered = True
        if self.verbose:
            print("[EnvClient] Registrado en TMInterface")

    def on_deregistered(self, iface: TMInterface):
        self._registered = False
        if self.verbose:
            print("[EnvClient] Desregistrado de TMInterface")

    def on_shutdown(self, iface: TMInterface):
        self._registered = False
        if self.verbose:
            print("[EnvClient] TMInterface se cerro")

    def on_simulation_begin(self, iface: TMInterface):
        self.initial_state = iface.get_simulation_state()
        iface.prevent_simulation_finish()
        if self.verbose:
            print("[EnvClient] Simulacion iniciada, estado inicial guardado")

    def on_simulation_step(self, iface: TMInterface, _time: int):
        try:
            action = self.action_queue.get(timeout=5.0)
        except queue.Empty:
            action = Action(accelerate=False, brake=False, steer=0)

        inputs = action_to_tminterface(action)
        iface.set_input_state(
            accelerate=inputs["accelerate"],
            brake=inputs["brake"],
            steer=inputs["steer"],
        )

        self._update_state(iface, _time)
        self.state_queue.put(self.state)
        self.step_event.set()

    def on_run_step(self, iface: TMInterface, _time: int):
        self.on_simulation_step(iface, _time)

    def on_simulation_end(self, iface: TMInterface, result: int):
        self.state.race_finished = True
        if self.verbose:
            print(f"[EnvClient] Simulacion terminada (result={result})")

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        pass

    def on_laps_count_changed(self, iface: TMInterface, current: int):
        pass

    def on_client_exception(self, iface: TMInterface, exception: Exception):
        print(f"[EnvClient] Excepcion: {exception}")

    def send_action(self, action: Action, timeout: float = 5.0) -> Optional[CarState]:
        self.step_event.clear()
        try:
            self.action_queue.put(action, timeout=timeout)
        except queue.Full:
            return None

        if self.step_event.wait(timeout=timeout * 2):
            try:
                return self.state_queue.get_nowait()
            except queue.Empty:
                return None
        return None

    def reset_to_initial(self):
        if self._iface and self.initial_state:
            self._iface.rewind_to_state(self.initial_state)

    def _update_state(self, iface: TMInterface, _time: int):
        try:
            sim_state = iface.get_simulation_state()
            self.state.time = _time
            self.state.position = list(sim_state.position)
            self.state.velocity = list(sim_state.velocity)
            self.state.yaw_pitch_roll = list(sim_state.yaw_pitch_roll)
            self.state.display_speed = sim_state.display_speed
            self.state.race_time = sim_state.race_time
            self.state.race_finished = sim_state.player_info.race_finished

            self.state.is_sliding = bool(sim_state.scene_mobil.is_sliding)
            self.state.engine_rpm = float(sim_state.scene_mobil.engine.actual_rpm)
            self.state.engine_gear = int(sim_state.scene_mobil.engine.gear)

            try:
                self.state.angular_speed = list(sim_state.dyna.angular_speed)
            except AttributeError:
                self.state.angular_speed = [0.0, 0.0, 0.0]

            self.state.input_steer = sim_state.input_steer
            self.state.input_gas = sim_state.input_gas
            self.state.input_accelerate = sim_state.input_accelerate
            self.state.input_brake = sim_state.input_brake

            wheels = sim_state.simulation_wheels
            self.state.wheel_ground_contact = [
                bool(w.real_time_state.has_ground_contact) for w in wheels
            ]

            self._step_count += 1
        except Exception as e:
            print(f"[EnvClient] Error leyendo estado: {e}")


class TrackmaniaEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        waypoints_path: str,
        client: Optional[EnvClient] = None,
        reward_config: Optional[RewardConfig] = None,
        max_steps: int = 3000,
        look_aheads: Optional[list] = None,
    ):
        super().__init__()

        self.max_steps = max_steps
        self.client = client

        self.wpm = WaypointManager(waypoints_path, look_aheads=look_aheads)
        self.state_extractor = StateExtractor(self.wpm)
        self.reward_config = reward_config or RewardConfig()
        self.reward_calc = RewardCalculator(self.reward_config, self.wpm)

        self.action_space = gymnasium.spaces.Discrete(NUM_ACTIONS)
        self.observation_space = self.state_extractor.observation_space

        self._current_waypoint_idx = 0
        self._total_steps = 0
        self._last_obs = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._current_waypoint_idx = 0
        self._total_steps = 0
        self.wpm.reset()
        self.reward_calc.reset()

        if self.client and self.client.initial_state:
            self.client.reset_to_initial()

        if self.client and self.client.state:
            car_state = self.client.state
            obs = self.state_extractor.extract(car_state, waypoint_idx=0)
        else:
            obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)

        self._last_obs = obs
        info = {"waypoint_idx": 0, "progress": 0, "speed": 0}
        return obs, info

    def step(self, action_idx: int):
        action = ACTIONS[action_idx]

        if self.client is None:
            obs = self._last_obs
            reward = 0.0
            terminated = True
            truncated = False
            info = {"waypoint_idx": 0, "progress": 0, "speed": 0, "term_reason": "no_client"}
            return obs, reward, terminated, truncated, info

        car_state = self.client.send_action(action)
        if car_state is None:
            obs = self._last_obs
            return obs, 0.0, True, False, {"waypoint_idx": 0, "progress": 0, "speed": 0, "term_reason": "client_timeout"}

        self._total_steps += 1

        reward, step_info, new_idx = self.reward_calc.compute(car_state, self._current_waypoint_idx)
        self._current_waypoint_idx = new_idx

        terminated, term_reason = self.reward_calc.should_terminate(
            car_state, new_idx, self._total_steps, self.max_steps
        )

        if terminated and term_reason == "finished":
            reward += self.reward_config.completion_bonus

        obs = self.state_extractor.extract(car_state, waypoint_idx=new_idx)
        self._last_obs = obs

        truncated = False
        step_info["term_reason"] = term_reason
        step_info["total_steps"] = self._total_steps

        return obs, reward, terminated, truncated, step_info

    def close(self):
        if self.client and self.client._registered and self.client._iface:
            self.client._iface.close()