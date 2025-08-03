# nmea2xyz

Utilities to parse NMEA and convert to Cartesian coordinates (ECEF --> centered --> rotationAligned).

## Install
```bash
pip install -e .[main]
```

## Usage
```bash
nmea2xyz -v --nmea tests/2025-06-28_18-44-12.nmea.txt --rotp 330 380 530
```
<img width="1078" height="520" alt="nmea2xyz" src="https://github.com/user-attachments/assets/30b9d5b5-c470-4c7c-ab27-1036cad65707" />

