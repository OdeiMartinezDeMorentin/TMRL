import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tminterface.client import run_client
from tmml.tmi.client import TMMLClient


def main():
    client = TMMLClient(verbose=True)
    print("[TMML] Conectando a TMInterface0...")
    print("[TMML] Asegurate de que Trackmania + TMInterface esten corriendo.")
    print("[TMML] Ctrl+C para detener.\n")
    run_client(client, server_name="TMInterface0")


if __name__ == "__main__":
    main()