# TMML - Trackmania Machine Learning

## Proyecto

Agente de Reinforcement Learning que aprende a conducir en Trackmania Nations Forever, conectado al juego mediante TMInterface v1.4.3 (API Python). Objetivo final: agente competitivo en pistas reales.

## Decisiones de Diseno

| Aspecto | Decision | Razon |
|---------|----------|-------|
| Juego | Trackmania Nations Forever | TMInterface proporciona acceso completo al estado |
| Interfaz | TMInterface v1.4.3 (Python API) | Mas simple para empezar; migrar a v2 si hace falta |
| Estado inicial | Telemetria pura (pos, vel, orientacion, ruedas) | MLP pequeno, inferencia rapida |
| Estado futuro | Evolucionar a LIDAR + telemetria | Mas informacion para curvas y pistas complejas |
| Acciones | Hibrido: accel/brake discretos, steering continuo | Trackmania se juega bien con teclado pero steering fino ayuda |
| Algoritmo inicial | DQN | Compatible con acciones discretas, bien probado en Linesight |
| Algoritmo futuro | SAC | Para acciones continuas, mayor sample efficiency |
| Pista inicial | Recta elevada sin muros (custom) | Deteccion clara de off-track via has_ground_contact |
| Recompensa | Progreso por waypoints + bonus completar + penalizacion off-track | Probada en TMRL y Linesight |
| Referencia | Linesight | Inspiracion arquitectonica, codigo original |

## Arquitectura de Conexion con TMInterface

```
Python (TMML)  <--shared memory mmap-->  TMInterface  <--injection-->  Trackmania
```

- Comunicacion via `mmap` (memory-mapped file `TMInterface0`)
- Python hereda de `Client` e implementa callbacks
- Cada tick (~10ms), TMInterface notifica al cliente
- El cliente lee estado, computa accion, envia input

### Flujo de un episodio

1. **`on_simulation_begin`**: Guardar estado inicial con `get_simulation_state()`
2. **`on_simulation_step(iface, time)`** (cada tick):
   - Leer estado: posicion, velocidad, orientacion, ruedas
   - Construir observacion normalizada
   - Seleccionar accion (epsilon-greedy en entrenamiento)
   - Enviar accion con `set_input_state()`
   - Calcular recompensa
   - Guardar transicion en replay buffer
3. **Condicion de terminacion**:
   - `has_ground_contact == False` en las 4 ruedas → se cayo (terminar inmediatamente)
   - Sin progreso en waypoints durante N pasos → failure countdown
   - Tiempo maximo excedido → timeout
   - `race_finished == True` → exito
4. **Fin de episodio**: guardar transicion final, rewind con `rewind_to_state(initial_state)`, entrenar sobre batch
5. Volver al paso 2

### Observaciones (Estado)

**Fase 1 - Telemetria pura (~20-30 features):**

| Feature | Fuente | Normalizacion |
|---------|--------|---------------|
| Velocidad (vx, vy, vz) | `state.velocity` | Dividir por velocidad max (~300 km/h) |
| Velocidad frontal | `state.display_speed` | Dividir por 300 |
| Orientacion (yaw, pitch, roll) | `state.yaw_pitch_roll` | Dividir por pi |
| Velocidad angular | `state.dyna.angular_speed` | Dividir por valor maximo |
| Contacto ruedas (4 bools) | `wheel.has_ground_contact` x4 | 0/1 |
| Derrape | `state.scene_mobil.is_sliding` | 0/1 |
| RPM motor | `state.scene_mobil.engine.actual_rpm` | Dividir por max_rpm |
| Engrane | `state.scene_mobil.gearbox_state` | Dividir por engrane max |
| Steer actual | `state.input_steer` | Dividir por 65536 |
| Gas actual | `state.input_gas` | Dividir por 65536 |
| Progreso pista | Waypoint actual / total | [0, 1] |
| Dist. al siguiente waypoint | Calculada desde posicion | Dividir por distancia max |

**Fase futura - LIDAR**: aadir 15-19 raycasts en 180 grados, procesados con red dedicada.

### Espacio de Acciones

```
Accion = { accelerate: bool, brake: bool, steer: float [-1, 1] }
```

- **DQN**: discretizar steering en 5 niveles (-1, -0.5, 0, 0.5, 1) → 2x2x5 = **20 acciones**
- **SAC futuro**: acciones puramente continuas, steer y gas analogicos

Mapeo a TMInterface:
```python
iface.set_input_state(
    accelerate=action.accelerate,
    brake=action.brake,
    steer=int(action.steer * 65536)
)
```

### Funcion de Recompensa

**Principal - Progreso por waypoints:**
1. Grabar trayectoria demo humana
2. Dividir en waypoints equidistantes
3. Recompensa = waypoints nuevos alcanzados desde el ultimo paso

**Adicionales:**
- Bonus grande por completar la pista: +100
- Penalizacion off-track: episodio termina, recompensa 0 por restante
- Terminacion temprana si sin progreso: failure countdown

**No usar velocidad como recompensa principal** (agente aprende a ir rapido sin frenar en curvas).

## Estructura del Proyecto

```
TMML/
├── agents.md
├── requirements.txt
├── config/
│   ├── default.yaml
│   └── tracks/
├── src/
│   ├── core/
│   │   ├── environment.py
│   │   ├── state.py
│   │   ├── actions.py
│   │   └── rewards.py
│   ├── models/
│   │   ├── dqn.py
│   │   ├── sac.py
│   │   └── networks.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── replay_buffer.py
│   │   └── logger.py
│   └── tmi/
│       └── client.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── collect_demo.py
└── tests/
    ├── test_env.py
    ├── test_actions.py
    ├── test_rewards.py
    └── test_training.py
```

## Fases de Desarrollo

### Fase 1 - MVP (conexion basica)

1. Set up TMInterface v1.4.3 + Trackmania Nations Forever
2. `tmi/client.py`: Wrapper de TMInterface
3. `core/environment.py`: Gymnasium env con step()/reset()
4. `core/state.py`: Extraccion y normalizacion de telemetria
5. `core/actions.py`: Espacio hibrido (20 acciones discretas)
6. `core/rewards.py`: Recompensa basica (progreso + off-track)
7. Pista de prueba: recta elevada sin muros en el editor
8. `collect_demo.py`: Grabar trayectoria demo
9. `models/networks.py`: MLP simple (3 capas, 128-256 neuronas)
10. `models/dqn.py`: DQN con replay buffer y target network
11. `training/trainer.py`: Loop de entrenamiento con logging
12. `tests/`: Smoke tests automaticos

**Hito**: Agente que acelera en linea recta sin caerse

**Validacion antes de entrenar largo**:
- Smoke test: agente random genera observaciones validas
- Overfit test: 30 min en situacion simple, verificar mejora
- Si smoke test falla → NO seguir, hay un bug

### Fase 2 - Aprender a conducir

1. Pista custom progresiva (recta → curva suave → curva cerrada → S)
2. Recompensa por progreso en waypoints (demo-based)
3. Normalizacion robusta con running statistics
4. Epsilon-greedy con decay
5. Target network update
6. Logging avanzado (reward/episodio, distancia, velocidad, %off-track)
7. Tunear hiperparametros

**Hito**: Agente que completa la pista custom consistentemente

### Fase 3 - Expansion

1. Observaciones LIDAR (raycasts)
2. SAC con acciones continuas
3. Comparar DQN vs SAC
4. Probar en pistas de campana reales
5. Considerar migrar a TMInterface v2 si necesario

**Hito**: Agente competitivo en pistas reales

## Dificultades y Riesgos

| Riesgo | Impacto | Mitigacion |
|--------|---------|------------|
| TMInterface v1 API inestable | Alto | Smoke tests; plan de migracion a v2 |
| Episodios atascados sin progreso | Medio | Failure countdown con terminacion temprana |
| Deriva numerica en rewinds | Medio | Re-guardar estado inicial cada N episodios |
| Inferencia lenta con modelos grandes | Medio | MLP pequeno primero; medir latencia |
| Recompensa mal formulada | Alto | Validacion progresiva; baseline random; overfit test |
| Overfitting a una pista | Medio | Randomizar estado inicial; entrenar en multiples pistas |

## Notas Tecnicas TMInterface

- `on_run_step(iface, time)`: llamado cada tick en modo carrera normal
- `on_simulation_step(iface, time)`: llamado cada tick en modo simulacion (sin render)
- `on_simulation_begin(iface)`: inicio de simulacion, guardar estado inicial
- `iface.set_speed(factor)`: ajustar velocidad del juego
- `iface.get_simulation_state()`: obtener estado completo del coche
- `iface.rewind_to_state(state)`: rebobinar a estado guardado (reset de episodio)
- `iface.set_input_state(accelerate=, brake=, steer=)`: enviar acciones
- `iface.prevent_simulation_finish()`: prevenir que la simulacion termine sola
- `iface.remove_state_validation()`: permitir modificar inputs durante validacion

## Hiperparametros Iniciales (a ajustar)

```yaml
dqn:
  learning_rate: 0.001
  batch_size: 64
  replay_buffer_size: 100000
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 10000
  target_update_freq: 1000
  hidden_layers: [128, 128]
  activation: relu

training:
  max_episodes: 10000
  max_steps_per_episode: 2000
  failure_countdown: 200
  eval_frequency: 100
  eval_episodes: 5
  save_frequency: 500

environment:
  step_frequency: 20
  reward_scale: 1.0
  completion_bonus: 100.0
```