**goals**
- build a light, not-too-bulky autonomous quadcopter on a custom flight-controller PCB (only ordering the board once — design carefully)
- deploy on a real drone i design — not a stock crazyflie as the end product
- learn how drones / FCs / IMUs / ESCs work, then get good at EE schematics + PCB layout before locking the board
- camera on it for recording video; maybe lidar later but not required for v1
- wireless control, ideally a handheld tilt-IMU remote (sticks are hard for me)
- leave room later for a companion computer if we run pedram-style learned control
- later: obstacle avoidance + some ML stuff
- keep cost low where possible; prefer smaller (~3") frame class

**constraints**
- one pcb spin if we can help it
- light / compact
- budget conscious
- real target = custom airframe + custom FC (sim is for learning / experiments only)

**near term**
- fundamentals: sensors, PID loops, ESC interface, power
- schematic skills → then layout (that part should be fun)
- pick size / power architecture that won't force a respin
- 3d-print a frame early so mechanical fit is real
- note findings here as i go
