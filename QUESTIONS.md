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

| 可能原因 | 你的代码中有什么证据 | 可能造成什么现象 |
|---|---|---|
| **观测没有归一化，训练逐渐发散** | actor 和 critic 的 `obs_normalization` 都是 `False` | 翻滚时出现很大的速度等输入，可能使网络逐渐失稳，最终出现 NaN；较晚的模型也可能表现变差 |
| **碰撞计算或物理状态失稳** | 仿真步长为 `0.005 s`；碰撞配置较硬，XML 中还有一个额外地板 | 特定姿态或撞击下可能出现异常速度、加速度，随后污染观测；需要捕获现场才能确认 |
| **执行器配置存在不一致** | XML 的 12 个力矩执行器仍保留，又添加了 12 个位置执行器；膝关节设定为 ±35.55 Nm，但关节总力矩仍受 XML 的 ±23.7 Nm 限制 | 实际控制效果可能与设计预期不同，需要核实额外执行器的控制值和实际输出 |
| **膝关节控制偏软，动作幅度又较大** | 膝关节位置增益约 **5.72 Nm/rad**；策略输出为 1 时，目标角度相对默认姿态偏移约 **1.55 rad（89°）** | 可能表现为腿弯下去、身体下沉，或探索动作过大；这只是嫌疑，不能仅凭数值判定 |
| **训练和播放测试难度较高** | 训练从粗糙地形开始，初始地形等级最高为 5，并且每 1–3 秒受到推力；播放也保留随机地形 | 模型可能还没有学会可靠站立；单次播放摔倒不足以判断它在平地上的能力 |

tensorboard

mkdir -p /root/tf-logs

ln -s /root/esra_unified_mjlab_simulator/logs/rsl_rl /root/tf-logs/leo
