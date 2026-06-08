import sys
import os
import json
import time
import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from tminterface.client import run_client, Client
from tminterface.interface import TMInterface
from tmml.core.waypoints import WaypointManager


class DemoCollector(Client):
    def __init__(self, track_name="default", verbose=True):
        super().__init__()
        self.track_name = track_name
        self.verbose = verbose
        self.positions = []
        self._collecting = True
        self._finished = False

    def on_registered(self, iface: TMInterface):
        if self.verbose:
            print(f"[Demo] Registrado en TMInterface")
            print(f"[Demo] Conduce la pista. Ctrl+C para detener y guardar.")

    def on_run_step(self, iface: TMInterface, _time: int):
        if not self._collecting:
            return
        try:
            sim_state = iface.get_simulation_state()
            pos = [float(sim_state.position[0]), float(sim_state.position[1]), float(sim_state.position[2])]
            self.positions.append(pos)
        except Exception as e:
            print(f"[Demo] Error: {e}")

    def on_simulation_step(self, iface: TMInterface, _time: int):
        self.on_run_step(iface, _time)

    def on_simulation_end(self, iface: TMInterface, result: int):
        self._finished = True
        if self.verbose:
            print(f"[Demo] Carrera terminada (result={result})")

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        if self.verbose:
            print(f"[Demo] Checkpoint: {current}/{target}")

    def on_laps_count_changed(self, iface: TMInterface, current: int):
        if self.verbose:
            print(f"[Demo] Lap: {current}")

    def save_raw(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        raw_path = os.path.join(output_dir, "raw_positions.json")
        data = {
            "track_name": self.track_name,
            "positions": self.positions,
            "count": len(self.positions),
        }
        with open(raw_path, "w") as f:
            json.dump(data, f, indent=2)
        if self.verbose:
            print(f"[Demo] Guardadas {len(self.positions)} posiciones en {raw_path}")
        return raw_path


def process_waypoints(raw_path: str, output_dir: str, track_name: str, spacing: float = 2.5, smooth_window: int = 50, smooth_iterations: int = 3):
    with open(raw_path, "r") as f:
        data = json.load(f)

    raw_positions = np.array(data["positions"], dtype=np.float64)
    print(f"[Process] Posiciones crudas: {len(raw_positions)}")

    smoothed = WaypointManager.smooth_waypoints(raw_positions, window=smooth_window, iterations=smooth_iterations)
    print(f"[Process] Tras suavizado: {len(smoothed)}")

    respaced = WaypointManager.respace_waypoints(smoothed, spacing=spacing)
    print(f"[Process] Tras reespaciado ({spacing}m): {len(respaced)} waypoints")

    diffs = np.diff(respaced, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    total_length = float(np.sum(segment_lengths))
    print(f"[Process] Longitud total: {total_length:.1f} metros")
    print(f"[Process] Espaciado medio: {np.mean(segment_lengths):.2f}m (min={np.min(segment_lengths):.2f}, max={np.max(segment_lengths):.2f})")

    wp_path = os.path.join(output_dir, "waypoints.json")
    WaypointManager.save_waypoints(wp_path, respaced, track_name=track_name, spacing=spacing)
    print(f"[Process] Waypoints guardados en {wp_path}")

    return wp_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Grabar demo y generar waypoints")
    parser.add_argument("--track", type=str, default="default", help="Nombre de la pista")
    parser.add_argument("--spacing", type=float, default=2.5, help="Espaciado entre waypoints (metros)")
    parser.add_argument("--smooth-window", type=int, default=50, help="Ventana de suavizado")
    parser.add_argument("--smooth-iterations", type=int, default=3, help="Iteraciones de suavizado")
    parser.add_argument("--skip-collect", action="store_true", help="Skip collection, process raw positions only")
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(__file__), "..", "config", "tracks", args.track)
    output_dir = os.path.abspath(output_dir)

    collector = DemoCollector(track_name=args.track)

    if not args.skip_collect:
        print("[Demo] Conectando a TMInterface0...")
        print("[Demo] Inicia la carrera y conduce la pista lentamente por el centro.")
        print("[Demo] Ctrl+C para detener y procesar waypoints.\n")

        try:
            run_client(collector, server_name="TMInterface0")
        except KeyboardInterrupt:
            pass

        collector.save_raw(output_dir)

    raw_path = os.path.join(output_dir, "raw_positions.json")
    if os.path.exists(raw_path):
        process_waypoints(
            raw_path, output_dir, args.track,
            spacing=args.spacing,
            smooth_window=args.smooth_window,
            smooth_iterations=args.smooth_iterations,
        )


if __name__ == "__main__":
    main()