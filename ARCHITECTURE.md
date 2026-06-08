# TMML - Referencia de Arquitectura

Descripcion detallada de cada archivo fuente: responsabilidad, contenido y relaciones.

---

## Core (`src/tmml/core/`)

### `environment.py`

**Responsabilidad**: Entorno compatible con Gymnasium que conecta el codigo RL en Python con Trackmania via TMInterface.

**Contenido**:
- `EnvClient(Client)`: Cliente de TMInterface thread-safe usando colas. Envia acciones y recibe `CarState` cada tick via `action_queue`/`state_queue` + `step_event`. Guarda el estado inicial para reseteo de episodio via `rewind_to_state`.
- `TrackmaniaEnv(gymnasium.Env)`: Entorno estandar Gymnasium. `reset()` rebobina al estado inicial, devuelve `(obs, info)`. `step(action_idx)` envia accion a TMInterface, recibe el siguiente estado, calcula recompensa y terminacion. Usa `StateExtractor`, `WaypointManager`, `RewardCalculator`.

**Relaciones**: Usa `StateExtractor`, `WaypointManager`, `RewardCalculator`, `Action`, `EnvClient`. Referenciado por `Trainer`, `scripts/train.py`, `scripts/evaluate.py`.

---

### `state.py`

**Responsabilidad**: Extraer y normalizar la telemetria cruda de `CarState` en un vector de observacion de tamano fijo para la red neuronal.

**Contenido**:
- `StateExtractor`: Recibe un `WaypointManager`. `observation_size` calcula dinamicamente la longitud del vector (26 con `look_aheads=[1,3,7]` por defecto). `observation_space` devuelve un `Box[-1, 1]`. `extract(car_state, waypoint_idx)` construye la observacion normalizada completa: velocidad (3), velocidad frontal (1), yaw/pitch/roll (3), velocidad angular (3), contacto ruedas (4), derrape (1), RPM (1), marcha (1), volante (1), acelerador (1), angulos lookahead (3), progreso (1), distancias lookahead (3).

**Relaciones**: Usa `WaypointManager` para angulos/distancias lookahead y progreso. Referenciado por `TrackmaniaEnv`.

---

### `actions.py`

**Responsabilidad**: Definir el espacio de acciones discreto para DQN.

**Contenido**:
- `Action` dataclass: `{accelerate: bool, brake: bool, steer: int}` con volante en `{-1, 0, 1}`.
- `ACTIONS`: Lista de 12 acciones (2 acelerar x 2 frenar x 3 volante).
- `NUM_ACTIONS = 12`, `ACTION_SPACE = Discrete(12)`.
- `action_to_tminterface()`: Convierte `Action` a diccionario de inputs de TMInterface (volante escalado por 65536).
- `tminterface_to_action()`: Conversion inversa.
- `get_action(index)` / `get_action_index(action)`: Helpers de busqueda.

**Relaciones**: Referenciado por `TrackmaniaEnv`, `DQNAgent` (usa `NUM_ACTIONS`).

---

### `rewards.py`

**Responsabilidad**: Calcular recompensas por paso y determinar condiciones de terminacion del episodio.

**Contenido**:
- `RewardConfig` dataclass: `progress_weight`, `speed_weight`, `step_penalty`, `completion_bonus`, `no_progress_timeout`.
- `RewardCalculator`: `compute(car_state, prev_idx)` devuelve `(recompensa, info, current_idx)`. Recompensa = progreso en waypoints * peso + bonus velocidad - penalizacion por paso. `should_terminate(car_state, current_idx, total_steps, max_steps)` devuelve `(terminated, reason)`. Comprueba: todas las ruedas fuera del suelo ("fallen"), carrera terminada ("finished"), sin progreso durante N pasos ("no_progress"), pasos maximos excedidos ("timeout"). `reset()` reinicia contadores internos.

**Relaciones**: Usa `WaypointManager` para `find_nearest`. Referenciado por `TrackmaniaEnv`.

---

### `waypoints.py`

**Responsabilidad**: Cargar, procesar y consultar waypoints a lo largo de una pista para seguimiento de progreso y lookaheads.

**Contenido**:
- `WaypointManager`: Carga waypoints desde JSON. `find_nearest(position, start_idx)` hace busqueda en ventana local. `get_lookahead_angles(position, yaw, idx)` devuelve angulos relativos normalizados a waypoints futuros. `get_lookahead_distances(position, idx)` devuelve distancias normalizadas a waypoints futuros. `progress_fraction(idx)` devuelve `[0, 1]`. `get_progress(position)` devuelve `(progreso, dist_al_siguiente)`. Metodos estaticos: `smooth_waypoints()`, `respace_waypoints()`, `load_waypoints()`, `save_waypoints()`.

**Relaciones**: Referenciado por `StateExtractor`, `RewardCalculator`, `collect_demo.py`.

---

## TMI (`src/tmml/tmi/`)

### `client.py`

**Responsabilidad**: Cliente de TMInterface de solo lectura para monitorizacion y recoleccion de demos.

**Contenido**:
- `CarState` dataclass: Almacena todos los campos de telemetria (posicion, velocidad, yaw_pitch_roll, display_speed, race_time, race_finished, wheel_ground_contact, is_sliding, angular_speed, engine_rpm, engine_gear, input_steer, input_gas, input_accelerate, input_brake).
- `TMMLClient(Client)`: Cliente pasivo que actualiza `CarState` cada tick. Usado para pruebas de conexion y grabacion de demos. NO envia acciones.

**Relaciones**: `EnvClient` en `environment.py` extiende `Client` por separado pero comparte `CarState`. Referenciado por `scripts/test_connection.py`, `scripts/collect_demo.py`.

---

## Models (`src/tmml/models/`)

### `networks.py`

**Responsabilidad**: Definir la arquitectura de la red neuronal (aproximador de funcion para los Q-valores).

**Contenido**:
- `QNetwork(nn.Module)`: MLP configurable. `__init__(obs_size, n_actions, hidden_layers=[128, 128], activation="relu")` construye las capas dinamicamente. `forward(x)` devuelve Q-valores para todas las acciones. Sin activacion en la capa de salida.

**Relaciones**: Usado por `DQNAgent` como `policy_net` y `target_net`.

---

### `dqn.py`

**Responsabilidad**: Orquestar el algoritmo DQN — seleccion de acciones, almacenamiento de experiencia, entrenamiento y persistencia del modelo.

**Contenido**:
- `DQNAgent`: `__init__(obs_size, n_actions, config)` crea redes policy/target, optimizador (Adam), perdida Huber y ReplayBuffer. `select_action(obs, epsilon)` usa epsilon-greedy. `store_transition(...)` agrega al replay buffer. `train_step(batch_size)` muestrea batch, calcula perdida DQN (target = recompensa + gamma * max(target_net(next_obs)) * (1-done)), backprop, clipping de gradientes. `update_target_network()` copia pesos de policy_net a target_net. `decay_epsilon()` reduce epsilon linealmente. `save(path)` / `load(path)` persisten checkpoint completo (pesos, optimizador, epsilon, config).

**Relaciones**: Usa `QNetwork`, `ReplayBuffer`, `DQNConfig`. Referenciado por `Trainer`.

---

## Training (`src/tmml/training/`)

### `replay_buffer.py`

**Responsabilidad**: Almacenar y muestrear transiciones de experiencia para aprendizaje off-policy.

**Contenido**:
- `ReplayBuffer`: Buffer circular de capacidad fija usando arrays numpy pre-asignados. `push(obs, action, reward, next_obs, done)` escribe en `pos` y da la vuelta. `sample(batch_size)` devuelve un batch aleatorio como arrays numpy. `__len__` devuelve el tamano actual.

**Relaciones**: Referenciado por `DQNAgent`.

---

### `logger.py`

**Responsabilidad**: Registrar y mostrar metricas de entrenamiento cada episodio y durante evaluaciones.

**Contenido**:
- `TrainingLogger`: Crea archivos CSV con timestamp en `log_dir/`. `log_episode(...)` escribe una fila por episodio (recompensa, pasos, epsilon, motivo de terminacion, waypoint, velocidad, progreso) e imprime por consola. `log_eval(...)` escribe metricas de evaluacion (recompensa media, pasos medios, tasa de finalizacion). `get_summary()` devuelve la media movil de recompensas recientes. `close()` limpia recursos.

**Relaciones**: Referenciado por `Trainer`.

---

### `trainer.py`

**Responsabilidad**: Loop principal de entrenamiento — el orquestador que conecta entorno, agente, logger y configuracion.

**Contenido**:
- `Trainer.__init__(env, agent, logger, config)`: Almacena referencias a todos los componentes.
- `train()`: Loop principal. Por cada episodio: `env.reset()`, loop de pasos con `agent.select_action()`, `env.step()`, `agent.store_transition()`, `agent.train_step()`. Despues de cada episodio: `agent.decay_epsilon()`, periodico `update_target_network()`, `logger.log_episode()`, periodico `evaluate()` y `save_checkpoint()`.
- `evaluate()`: Ejecuta N episodios con epsilon=0, registra metricas.
- `save_checkpoint(episode)`: Guarda modelo en `checkpoint_dir/`.

**Relaciones**: Usa `TrackmaniaEnv`, `DQNAgent`, `TrainingLogger`, `TMMLConfig`.

---

## Config (`src/tmml/config/`)

### `loader.py`

**Responsabilidad**: Parsear la configuracion YAML y crear dataclasses tipadas para todos los componentes.

**Contenido**:
- `DQNConfig` dataclass: lr, batch_size, replay_buffer_size, gamma, schedule de epsilon, target_update_freq, hidden_layers, activation.
- `TrainingConfig` dataclass: max_episodes, max_steps, failure_countdown, frecuencia eval/save, directorios log/checkpoint.
- `EnvironmentConfig` dataclass: waypoints_path, max_steps, look_aheads, `RewardConfig` anidado.
- `TMMLConfig` dataclass: combina `DQNConfig`, `TrainingConfig`, `EnvironmentConfig`.
- `load_config(path)`: Lee YAML, mapea secciones a dataclasses con valores por defecto.

**Relaciones**: Referenciado por `scripts/train.py`, `scripts/evaluate.py`. Los valores cargados fluyen hacia todos los componentes.

---

## Scripts (`scripts/`)

### `train.py`

**Responsabilidad**: Punto de entrada CLI que conecta todo e inicia el entrenamiento.

**Contenido**: Parsea argumentos (`--config`, `--episodes`, `--lr`, `--epsilon-start`), carga configuracion, crea `EnvClient` + hilo de conexion, espera conexion con TMInterface, crea `TrackmaniaEnv`, `DQNAgent`, `TrainingLogger`, `Trainer`, y llama a `trainer.train()`. Maneja Ctrl+C correctamente.

**Relaciones**: Usa todos los componentes principales.

---

### `evaluate.py`

**Responsabilidad**: Cargar un modelo entrenado y ejecutar episodios de evaluacion (epsilon=0) para medir rendimiento.

**Contenido**: Parsea argumentos (`--model`, `--config`, `--episodes`), se conecta a TMInterface, carga modelo en `DQNAgent`, ejecuta episodios, imprime estadisticas resumen (recompensa media, completaciones).

**Relaciones**: Usa `TrackmaniaEnv`, `DQNAgent`, `load_config`.

---

### `collect_demo.py`

**Responsabilidad**: Grabar posiciones de conduccion humana desde Trackmania y procesarlas en waypoints.

**Contenido**: `DemoCollector(Client)` registra la posicion del coche cada tick. Post-procesa con `WaypointManager.smooth_waypoints()` y `respace_waypoints()`, guarda el JSON final.

**Relaciones**: Usa `TMMLClient`, `WaypointManager`.

---

### `test_connection.py`

**Responsabilidad**: Script minimo para verificar que la conexion con TMInterface funciona.

**Contenido**: Crea `TMMLClient(verbose=True)` y ejecuta `run_client()`.

---

### `plot_waypoints.py`

**Responsabilidad**: Visualizar waypoints desde un archivo JSON (vista superior + perfil de elevacion).

**Contenido**: Lee waypoints, dibuja X vs Z y elevacion, guarda `waypoints_plot.png`.

---

## Tests (`tests/`)

### `test_actions.py`

Tests para `actions.py`: verifica 12 acciones, valores de volante, conversion de ida y vuelta, indices de accion.

### `test_rewards.py`

Tests para `RewardCalculator`: recompensa por progreso, penalizacion por paso, bonus de velocidad, condiciones de terminacion (caida, finalizado, sin progreso, timeout), reseteo.

### `test_waypoints.py`

Tests para `WaypointManager`: find_nearest, progress_fraction, smooth, respace, angulos/distancias lookahead, reseteo.

### `test_env.py`

Tests para `TrackmaniaEnv` con cliente mock: forma de observacion, reset/step sin cliente, step con mock, terminacion por caida/timeout, tipo del espacio de acciones.

### `test_training.py`

Tests para `ReplayBuffer` y `DQNAgent`: push/sample, desbordamiento, seleccion de acciones epsilon-greedy, calculo de perdida en train_step, actualizacion de target network, decaimiento de epsilon, guardar/cargar checkpoint.