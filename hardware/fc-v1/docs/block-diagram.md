# FC v1 block diagram

```
Battery ──► Power input + filter caps ──► 5V reg ──► 3.3V reg ──► MCU + IMU
                                      │
                                      └──► ESC power (high current) ──► 4-in-1 ESC (external) ──► motors

MCU ◄── SPI/I2C ──► IMU (gyro+accel)
MCU ◄── UART ────► Radio RX (ELRS/CRSF)
MCU ◄── PWM/DSHOT ► ESC signal connector
MCU ◄── SWD ─────► debug header
MCU ◄── UART ────► future Pi Zero (optional header)
```

Keep IMU away from ESC/motor current paths. Soft-mount board or isolate IMU mechanically.
