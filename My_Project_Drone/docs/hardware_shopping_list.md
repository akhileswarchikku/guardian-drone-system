# Hardware Shopping List — EDA/GSR Sensor

For real EDA data to replace imputed features in the Apple Watch adapter.

## Buy This (DIY — ~$15 total)

| Item | Where to Buy | Price |
|------|-------------|-------|
| ESP32 Dev Board (or Arduino Nano) | Amazon / AliExpress | ~$5 |
| Grove GSR Sensor (Seeed Studio) | Amazon / SeeedStudio.com | ~$8 |
| Jumper wires + USB cable | Included or ~$2 | ~$2 |

Search terms:
- Amazon: "Grove GSR sensor" or "ESP32 GSR skin conductance"
- AliExpress: "Grove galvanic skin response sensor"

## Why These

The Grove GSR sensor measures skin conductance (EDA) continuously.
Combined with ESP32 Bluetooth, it streams real EDA data to the laptop.
This is the same signal as the $2,000 Empatica E4 used in the WESAD dataset.

## What NOT to Buy

- Empatica E4 — research only, $2,000, needs institutional license
- Shimmer3 GSR+ — research only, $600+
- Muse S — measures EEG (brain), NOT EDA (skin) — wrong sensor
- Fitbit Sense 2 — EDA only in 2-min manual scans, not continuous

## Next Steps After Purchase

1. Flash ESP32 with Arduino sketch (ask Claude Code to generate it)
2. Wear Grove sensor on 2 fingers while Apple Watch on wrist
3. Record 30-min session (rest + stress activities)
4. Feed CSV into biometric_ml/apple_watch_adapter.py
5. Get real EDA-based predictions
