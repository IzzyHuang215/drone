**goals**
- build a light, not-too-bulky autonomous quadcopter (custom pcb as the base — only ordering the board once so design carefully)
- deploy whatever we learn (pid, rl, etc.) on a real drone i design — not a stock crazyflie as the end product
- start by learning how drones / flight controllers / IMUs / ESCs actually work before locking the hardware
- camera on it for recording video; maybe lidar later but not required for v1
- wireless control, ideally a handheld controller with an IMU so tilting it makes the drone go that way (sticks are hard for me)
- later: obstacle avoidance + some ML stuff
- keep cost low where possible; prefer smaller frame class so parts + crashes are cheaper

**constraints**
- one pcb spin if we can help it
- light / compact
- budget conscious
- real target platform = custom design (sim / crazyflie-like models are for learning / experiments only)

**near term**
- fundamentals + resources
- pick a first size / architecture that won't force a board respin
- clarify with pedram how his crazyflow / ppo work maps onto the custom build (dynamics model, what transfers)
- note findings here as i go
