# TMRL

Proyecto en Python para desarrollar un agente de aprendizaje por refuerzo capaz de conducir en Trackmania usando TMInterface. El repositorio está orientado a construir un entorno de entrenamiento basado en telemetría, waypoints y recompensas para entrenar modelos tipo DQN sobre escenarios de conducción.

## Descripción general

TMRL combina varios componentes:

- conexión con TMInterface para leer el estado del vehículo y enviar acciones,
- extracción y normalización de observaciones,
- cálculo de recompensas por progreso y penalizaciones,
- gestión de waypoints para seguir la pista,
- entrenamiento de políticas de RL con PyTorch,
- scripts de validación, evaluación y captura de demos.

La estructura del código está organizada en módulos bajo `src/tmml`, con pruebas automáticas en `tests` y utilidades de ejecución en `scripts`.

## Características principales

- Entorno compatible con Gymnasium.
- Estado del coche basado en velocidad, orientación, contacto de ruedas, derrape, RPM y progreso en la pista.
- Sistema de acciones híbrido: aceleración, frenado y dirección.
- Cálculo de recompensas por progreso de waypoint, bonus de finalización y penalizaciones por salida de pista o falta de progreso.
- Estructura preparada para pruebas de DQN y extensiones futuras con políticas continuas.
- Soporte para captura de demos humanas y generación de waypoints.

## Stack tecnológico

- Python 3.10+
- PyTorch
- Gymnasium
- NumPy
- TMInterface
- PyYAML
- Matplotlib

## Estructura del repositorio

```text
TMRL/
├── ARCHITECTURE.md            # Descripción detallada de la arquitectura del proyecto
├── agents.md                  # Documento central de diseño del agente y decisiones técnicas
├── requirements.txt           # Dependencias del proyecto
├── pyproject.toml             # Configuración del paquete y herramientas de Python
├── config/                    # Configuraciones y pistas
├── scripts/                   # Entradas de ejecución: entrenamiento, evaluación y demos
├── src/
│   └── tmml/
│       ├── config/
│       ├── core/
│       ├── models/
│       ├── tmi/
│       └── training/
├── tests/                     # Pruebas unitarias y de integración ligera
├── .gitignore
└── README.md
```

## Instalación

1. Clona el repositorio:

```bash
git clone https://github.com/OdeiMartinezDeMorentin/TMRL.git
cd TMRL
```

2. Crea un entorno virtual e instálalo:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Requisitos de ejecución

Este proyecto está pensado para ejecutarse junto con:

- Trackmania Nations Forever
- TMInterface
- una pista de prueba o una configuración de waypoints adecuada

La conexión con el juego se realiza mediante TMInterface, que comunica el estado del coche con el entorno Python.

## Uso rápido

### Ejecutar pruebas

```bash
pytest
```

### Entrenar el agente

El flujo principal se gestiona desde `scripts/` y se conecta con la lógica del entorno y el agente.

```bash
python scripts/train.py
```

### Evaluar un modelo entrenado

```bash
python scripts/evaluate.py --model <ruta_al_modelo>
```

### Capturar una demo

```bash
python scripts/collect_demo.py
```

## Arquitectura conceptual

El proyecto sigue una arquitectura típica de RL:

```text
Trackmania + TMInterface
        ↓
TMML client / env
        ↓
Observaciones (telemetría + waypoints)
        ↓
Agente RL (DQN / extensible a SAC)
        ↓
Acciones (acelerar, frenar, girar)
        ↓
Recompensa y fin de episodio
```

La implementación actual se centra en:

- aprender desde una representación estructurada del estado del vehículo,
- maximizar el progreso por pista,
- evitar caídas y pérdidas de tracción,
- mantener un entorno de entrenamiento reproducible y testeable.

## Documentación incluida

Además del README, el repositorio incluye los siguientes documentos:

- `ARCHITECTURE.md`: referencia detallada del diseño por módulos.
- `agents.md`: descripción del proyecto, decisiones técnicas y plan de desarrollo.

## Estado de desarrollo

El proyecto está orientado a un MVP de aprendizaje por refuerzo en Trackmania con una base de observación y recompensas funcionales. El enfoque documentado contempla una progresión desde telemetría básica hacia escenarios más complejos y modelos más avanzados.

## Licencia

No se especifica licencia en el repositorio en esta fase inicial. Si necesitas añadir una licencia, puedes decidir entre MIT, Apache 2.0 o GPL según el uso que le vayas a dar al proyecto.

## Contribución

Las contribuciones son bienvenidas. Para colaborar:

1. Haz un fork del repositorio.
2. Crea una rama para tu cambio.
3. Añade pruebas relevantes.
4. Abre un pull request con una descripción clara del cambio.

## Contacto

Repositorio: https://github.com/OdeiMartinezDeMorentin/TMRL

Si quieres, puedo dejarte también una versión del README más breve, más técnica o más orientada a GitHub con badges y una estética más profesional.
