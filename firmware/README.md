# firmware/ — Teensy 4.1 firmware

Multi-channel sEMG signal acquisition. Folders are organized by date; **the latest is `0623SDcard/`**.

| Folder | Description |
|---|---|
| `0505/V0/V0.ino` | Early version: Serial output + SD card writing |
| `0623SDcard/ReadSD/ReadSD.ino` | **Current version**: SD card reading |

## Hardware

* **Microcontroller**: Teensy 4.1
* **Sensors**: Grove EMG Sensor ×2
  * `Raw0` → main muscle
  * `Raw1` → synergist
* **Sampling rate**: 1000 Hz
* **Communication**: Serial; see the top of each `.ino` file for the baud rate and pin settings

## Dependencies (not included in this repo)

The SD card MTP feature needs the third-party library **MTP_Teensy** (about 24MB, so it is not version-controlled):

* Download: <https://github.com/KurtE/MTP_Teensy>
* Unzip it into Arduino's `libraries/` directory

## Data flow

```
Teensy 4.1 (1000Hz) ──> SD card / Serial ──> Dataset/Raw/*_Raw_DATA.csv
                                                │
                                                ▼
                                pipeline/emg_process_segment_FIXED.py
```

The output CSV uses `Time_ms == 1` as the segment start marker, and the preprocessing program splits segments accordingly.

> ⚠️ If the EMG device is accidentally pressed twice during recording, a very short empty segment is produced,
> and the OpenCap video does not stop with it — so that take's TRC has extra time at its start.
> In that case the head of the TRC must be trimmed, and `segment_ids` in `Features_insert_FIXED.py` updated to match.
> (This actually happened once; see methodological pitfall 4 in `pipeline/README.md`.)
