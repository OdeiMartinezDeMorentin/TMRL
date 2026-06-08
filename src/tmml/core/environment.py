import threading
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
    NO_ACTION = Action(accelerate=False, brake=False, steer=0)

    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.state = CarState()
        self._registered = False
        self._step_count = 0

        self._action_lock = threading.Lock()
        self._pending_action = None
        self._last_action = self.NO_ACTION
        self._pending_reset = False

        self._new_state = threading.Event()
        self._state_lock = threading.Lock()
        self._latest_car_state = None

        self.initial_state = None
        self._iface = None

    def on_registered(self, iface: TMInterface):
        self._iface = iface
        self._registered = True
        iface.set_timeout(-1)
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

    def _ensure_initial_state(self, iface: TMInterface):
        if self.initial_state is None:
            self.initial_state = iface.get_simulation_state()
            if self.verbose:
                print("[EnvClient] Estado inicial capturado")

    def on_simulation_begin(self, iface: TMInterface):
        self._ensure_initial_state(iface)
        iface.prevent_simulation_finish()
        with self._action_lock:
            self._pending_action = None
            self._last_action = self.NO_ACTION
        self._new_state.clear()
        if self.verbose:
            print("[EnvClient] Simulacion iniciada")

    def _get_action(self):
        with self._action_lock:
            action = self._pending_action
            if action is not None:
                self._pending_action = None
                self._last_action = action
            else:
                action = self._last_action
        return action

    def on_simulation_step(self, iface: TMInterface, _time: int):
        self._ensure_initial_state(iface)

        with self._action_lock:
            reset_needed = self._pending_reset

        if reset_needed and self.initial_state:
            with self._action_lock:
                self._pending_reset = False
                self._pending_action = None
                self._last_action = self.NO_ACTION
            iface.rewind_to_state(self.initial_state)
            if self.verbose:
                print("[EnvClient] Rewind ejecutado")
            return

        action = self._get_action()

        inputs = action_to_tminterface(action)
        iface.set_input_state(
            accelerate=inputs["accelerate"],
            brake=inputs["brake"],
            steer=inputs["steer"],
        )

        self._update_state(iface, _time)
        with self._state_lock:
            self._latest_car_state = self.state
        self._new_state.set()

    def on_run_step(self, iface: TMInterface, _time: int):
        self.on_simulation_step(iface, _time)

    def on_simulation_end(self, iface: TMInterface, result: int):
        self.state.race_finished = True
        with self._state_lock:
            self._latest_car_state = self.state
        self._new_state.set()
        if self.verbose:
            print(f"[EnvClient] Simulacion terminada (result={result})")

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        pass

    def on_laps_count_changed(self, iface: TMInterface, current: int):
        pass

    def on_client_exception(self, iface: TMInterface, exception: Exception):
        print(f"[EnvClient] Excepcion: {exception}")

    def send_action(self, action: Action, timeout: float = 10.0) -> Optional[CarState]:
        self._new_state.clear()
        with self._action_lock:
            self._pending_action = action
            self._pending_reset = False

        if self._new_state.wait(timeout=timeout):
            with self._state_lock:
                return self._latest_car_state
        return None

    def reset_to_initial(self, timeout: float = 10.0) -> Optional[CarState]:
        if not self.initial_state:
            return None

        if self.verbose:
            print(f"[EnvClient] reset_to_initial llamado (pos actual: {[f'{v:.1f}' for v in self.state.position]})")

        with self._action_lock:
            self._pending_reset = True
            self._pending_action = None
            self._last_action = self.NO_ACTION

        self._new_state.clear()

        if self._new_state.wait(timeout=timeout):
            with self._state_lock:
                result = self._latest_car_state
            if self.verbose:
                print(f"[EnvClient] reset completado (pos: {[f'{v:.1f}' for v in result.position]}, wheels: {result.wheel_ground_contact})")
            return result

        if self.verbose:
            print("[EnvClient] reset_to_initial TIMEOUT")
        return None

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

        car_state = None
        if self.client and self.client.initial_state:
            car_state = self.client.reset_to_initial()

        if car_state is None and self.client and self.client.state:
            car_state = self.client.state

        if car_state is not None:
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