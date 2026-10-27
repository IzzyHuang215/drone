# Learning-based autonomous drone: conversation notes

Source: [Build Autonomous Drone](https://chatgpt.com/c/6ade8fae-7b88-83ea-a3f3-7c5415f6abd4), discussed October 25, 2026; imported October 26, 2026. The source conversation may require account access.

These notes summarize the visible conversation, including the replacement answer after the original project plan was cut off. Recommendations, prices, simulator capabilities, and research claims below are recorded from that discussion, not independently verified specifications or purchasing advice. No hardware purchase, implemented controller, flight result, or proven theoretical result was established in the conversation.

## User goals and scope

- Build an autonomous drone with learning-based control and reinforcement learning as the main interests.
- Modify an existing drone, with a budget of **$300–700 USD**.
- First priority: **“fly really well to start with.”** Focus on precise hover, trajectory tracking, disturbance rejection, and eventually agile maneuvers.
- Navigation, computer vision, exploration, and obstacle avoidance are later directions.

The existing [goals note](goals.md) describes a custom PCB base, compact construction, a camera, and possible IMU-based handheld control. The conversation instead explored an existing research drone. These are separate candidate approaches; the discussion did not resolve whether to choose one or combine them. Preserve the one-PCB-spin constraint if the custom-board approach proceeds.

## Hardware candidates discussed

The initial recommendation was the **Bitcraze Crazyflie 2.1+**. After clarifying the interest in agile flight, the assistant suggested investigating the **Crazyflie Brushless** as well.

| Item | Estimate quoted in the conversation | Role / tradeoff |
| --- | --- | --- |
| Crazyflie 2.1+ | ~$240 | Brushed motors; more room in the budget for initial control experiments |
| Crazyflie Brushless starter kit | ~$500 | Suggested as a stronger long-term candidate for demanding maneuvers; less budget flexibility |
| Flow Deck v2 | ~$55 | Optical-flow sensing |
| Crazyradio 2.0 | ~$43 | Communication with a laptop |
| Spare batteries, propellers, and guards | ~$40–70 | Flight sessions and repairs |

The discussion estimated **$380–450** for the standard platform and accessories before extra localization equipment. This was an approximate historical estimate, not a verified bill of materials; check current prices, kit contents, compatibility, shipping, and localization needs before ordering.

**Localization remains unresolved:** the Flow Deck alone does not provide precise global position measurements. Rigorous real-world trajectory tracking needs an explicit localization and ground-truth measurement plan.

## Software and control architecture discussed

- Python and PyTorch.
- Gymnasium with a drone simulator.
- Stable-Baselines3 for PPO / SAC experiments.
- Bitcraze's Python library for communication with the physical drone.
- **Crazyflow** as a simulator candidate. The conversation described it as a JAX-based differentiable simulator with identified Crazyflie dynamics, including brushless models, and sim-to-real demonstrations. Its actual platform support and interfaces still need checking before adoption; the discussion did not establish that it plugs directly into the PyTorch / Stable-Baselines3 stack.

Two levels of learned control were distinguished:

| Level | Policy output | Role of existing firmware |
| --- | --- | --- |
| High-level RL | Desired velocity or attitude | Retains onboard stabilization; suggested as the easier initial transfer path |
| Lower-level learned control | Body-rate and thrust commands, potentially individual motor commands later | Requires progressively more control integration and validation |

Low-level control was presented as a possible eventual research target, not an agreed first implementation. Initial physical transfer should retain onboard stabilization, command limits, and an independent emergency stop.

## Proposed progression

The opening response displayed an **8–12 week** timeline, but its detailed roadmap was truncated. The later visible replies supplied these phases, without a committed schedule:

1. **Simulation and baseline:** model or simulate the drone; implement a classical PID controller and compare with LQR where appropriate. Before buying hardware, train a first simulated hovering policy to assess interest and identify hardware requirements.
2. **Learn to hover:** use PPO with randomized initial positions, velocities, and attitudes; compare against classical baselines.
3. **Trajectory tracking:** progress through circles, figure-eights, and increasingly demanding reference trajectories.
4. **Disturbance rejection and robustness:** randomize mass, inertia, thrust response, and aerodynamic drag; introduce wind gusts and sensor noise.
5. **Physical transfer:** validate through software-in-the-loop tests, then protected, controlled flights with command limits and an independent emergency stop. Resolve localization before claiming precise physical tracking.
6. **Advanced experiments:** residual RL, model-based RL, online system identification, and adaptive policies. One suggestion was a learned residual that supplements MPC to compensate for model errors.

## Connection to ESE 6180 generative flows

The user asked whether [ese6180-generative-flows](https://github.com/pedrambayat/ese6180-generative-flows) relates. The assistant connected the proposal *Generalization and Stability of Generative Flows Under Feedback* to learned drone dynamics and feedback control.

The common research question is whether feedback improves stability while shifting the system into states where a learned model is unreliable.

| Generative-flow concept | Drone-control analogue proposed in the discussion |
| --- | --- |
| Learned vector field | Learned dynamics model |
| Inference-time guidance | Feedback controller |
| Training-path distribution | States observed during training |
| Guided rollout distribution | States visited during flight |
| Incremental stability | Sensitivity to disturbances and initial errors |
| Terminal distribution error | Trajectory or model prediction error |

The discussed formulations were

$$
\dot{x}_t = v_\theta(x_t,t) + u_t(x_t)
$$

for a guided generative flow, versus

$$
\dot{x}_t = f_\theta(x_t,u_t) + w_t
$$

for learned drone dynamics with disturbances. Sampling time and physical time are different, and physical control enters through the dynamics rather than necessarily as an additive vector field. The analogy is not a direct transfer of a theorem.

The conversation quoted a perturbation amplification factor and a coverage coefficient:

$$
\Gamma_u(1,t) = \exp\!\left(\int_t^1 \ell_u(s)\,ds\right),
\qquad
C_t^u = \left\|\frac{dq_t^u}{d\mu_t}\right\|_\infty.
$$

It described the proposal's target bound as

$$
W_2(\hat{p}_1^u,p_1^u)
\le E_n\left(\int_0^1 \Gamma_u(1,t)^2 C_t^u\,dt\right)^{1/2}
+ \epsilon_{\mathrm{solver}}.
$$

This is recorded as a **proposed result from the conversation**, not a verified or proven bound. Informally, the suggested drone analogy was prediction error depending on model error, closed-loop sensitivity, and a coverage penalty. A drone-specific result would require its own assumptions and derivation.

### Suggested experiment

Compare three controllers under increasingly difficult trajectories and mismatched physical parameters:

1. Classical MPC using an approximate physics model.
2. Learned-model MPC using a neural dynamics model trained on simulated or real trajectories.
3. Coverage-aware learned-model MPC discouraging poorly supported state-action regions while tracking the reference.

Hypothesis: accounting for training coverage may trade some nominal tracking performance for better generalization under unseen dynamics. This is an experimental hypothesis, not a demonstrated outcome.

Suggested measurements:

- Tracking root-mean-square error.
- Perturbation amplification.
- Prediction error on visited state-action pairs.
- Coverage relative to training data.
- Failure rate under randomized disturbances.

### Longer-term generative controller

An optional later direction was a conditional flow model generating candidate action sequences

$$
(u_t,u_{t+1},\ldots,u_{t+H}) \sim p_\theta(\cdot\mid x_t).
$$

Proposed pipeline: current state → conditional generative flow → candidate action trajectories → feedback guidance / trajectory scoring → feasible action selection → drone with a stabilizing flight controller.

Guidance could favor tracking accuracy, low control effort, and feasible flight. The research question is whether stronger guidance improves generated trajectories or pushes them beyond the model's training coverage. The conversation regarded this as more ambitious than PPO or conventional MPC.

## Decisions still needed

- Existing Crazyflie platform versus custom PCB / custom quadcopter route.
- Standard versus brushless hardware, with a complete and current budget.
- Simulator, learning framework, and supported control interfaces.
- Localization and ground-truth tracking setup.
- First policy's action level and physical transfer criteria.
- Engineering / learning project versus a research project aimed at publication; the assistant asked this, but the visible conversation contains no answer.

The final recommendation in the conversation was to establish conventional flight control first, then explore learned-model MPC and stability / coverage questions. Flow-based planning was a later direction. It also advised keeping the ESE 6180 synthetic theory experiments manageable, with the drone as possible longer-term physical validation.
