# FC v1 — custom flight controller (KiCad)

Board is the drone center base: MCU + IMU + power + ESC interface + RX + SWD.
Pi Zero is a later plug-in (UART + 5V header), not on this schematic yet.

## Open in KiCad
1. Install KiCad from https://www.kicad.org/ (or winget: `winget install KiCad.KiCad`)
2. File → Open Project → `fc-v1.kicad_pro`

## Design order
1. Fix outline + mounting holes (match 3D-printed frame)
2. Block diagram (done in docs/block-diagram.md)
3. Pick MPNs → add symbols/footprints to `libs/`
4. Schematic → ERC
5. Layout → DRC → Gerbers

## Board intent (v1)
- Outline: start ~30×30 mm or 40×40 mm (adjust after frame print)
- Mounting: M2 or M3 holes at corners (match stack pattern)
- External 4-in-1 ESC connector preferred for first spin (ESC on-board later)
