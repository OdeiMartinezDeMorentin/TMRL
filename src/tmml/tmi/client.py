import time
from dataclasses import dataclass, field
from tminterface.client import Client
from tminterface.interface import TMInterface


@dataclass
class CarState:
    time: int = -1
    position: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    yaw_pitch_roll: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    display_speed: int = 0
    race_time: int = 0
    race_finished: bool = False
    wheel_ground_contact: list = field(default_factory=lambda: [False, False, False, False])
    is_sliding: bool = False
    angular_speed: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    engine_rpm: float = 0.0
    engine_gear: int = 0
    input_steer: int = 0
    input_gas: int = 0
    input_accelerate: bool = False
    input_brake: bool = False

    def __str__(self):
        lines = [
            f"CarState(time={self.time})",
            f"  pos=({self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f})",
            f"  vel=({self.velocity[0]:.2f}, {self.velocity[1]:.2f}, {self.velocity[2]:.2f})",
            f"  ypr=({self.yaw_pitch_roll[0]:.4f}, {self.yaw_pitch_roll[1]:.4f}, {self.yaw_pitch_roll[2]:.4f})",
            f"  speed={self.display_speed} km/h",
            f"  finished={self.race_finished}",
            f"  wheels_on_ground={self.wheel_ground_contact}",
            f"  sliding={self.is_sliding}",
            f"  rpm={self.engine_rpm:.0f}  gear={self.engine_gear}",
            f"  angular=({self.angular_speed[0]:.2f}, {self.angular_speed[1]:.2f}, {self.angular_speed[2]:.2f})",
            f"  steer={self.input_steer}  gas={self.input_gas}",
            f"  accel={self.input_accelerate}  brake={self.input_brake}",
        ]
        return "\n".join(lines)


class TMMLClient(Client):
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.state = CarState()
        self._registered = False
        self._step_count = 0

    def on_registered(self, iface: TMInterface):
        self._registered = True
        iface.set_timeout(-1)
        if self.verbose:
            print("[TMML] Registrado en TMInterface")

    def on_deregistered(self, iface: TMInterface):
        self._registered = False
        if self.verbose:
            print("[TMML] Desregistrado de TMInterface")

    def on_shutdown(self, iface: TMInterface):
        self._registered = False
        if self.verbose:
            print("[TMML] TMInterface se cerro")

    def on_run_step(self, iface: TMInterface, _time: int):
        self._update_state(iface, _time)

    def on_simulation_step(self, iface: TMInterface, _time: int):
        self._update_state(iface, _time)

    def on_simulation_begin(self, iface: TMInterface):
        if self.verbose:
            print("[TMML] Simulacion iniciada")

    def on_simulation_end(self, iface: TMInterface, result: int):
        if self.verbose:
            print(f"[TMML] Simulacion terminada (result={result})")

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        if self.verbose:
            print(f"[TMML] Checkpoint: {current}/{target}")

    def on_laps_count_changed(self, iface: TMInterface, current: int):
        if self.verbose:
            print(f"[TMML] Lap: {current}")

    def on_client_exception(self, iface: TMInterface, exception: Exception):
        print(f"[TMML] Excepcion: {exception}")

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
            if self.verbose and self._step_count % 200 == 0:
                print(self.state)

        except Exception as e:
            print(f"[TMML] Error leyendo estado: {e}")