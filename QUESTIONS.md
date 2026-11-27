| Item | `leo.xml` | `leo_constants.py` | What is actually used |
|---|---|---|---|
| Maximum knee torque | ±23.7 Nm | ±35.55 Nm | **Training: still capped at the XML limit of ±23.7 Nm**; manual viewer: ±35.55 Nm from constants |
| Reflected rotor inertia (kg·m²) | 0.002 for every joint | Hip ≈0.004026; knee ≈0.001449 | **Training and manual viewer: constants** |
| Initial height | 0.42 m | 0.30 m | **Training and manual viewer: constants**; training adds a random height offset |
| Initial joint angles | All zero | Hip roll: 0; hip pitch: 1.0; knee pitch: −1.66 rad | **Training and manual viewer: constants** |
| Floor | Already includes a floor | Comment says the XML has no floor; running this file directly adds another floor | **Training and manual viewer: the XML floor**; running constants directly adds a duplicate floor |
| Self-collision | Exclusion list is commented out; collision bitmasks also prevent self-collision | Comment incorrectly says the XML exclusions are active; separate collision bitmasks prevent self-collision | **Training: collision bitmasks from constants**; manual viewer: collision bitmasks from XML |
| Actuators | Includes 12 torque actuators | Adds 12 position actuators | **Training: both sets remain, totaling 24; the policy controls the 12 added position actuators**; manual viewer: the 12 XML actuators |

The current code does not account for the following safety considerations:

- Monitor voltage and temperature.
- Avoid collisions between the robot's limbs.
