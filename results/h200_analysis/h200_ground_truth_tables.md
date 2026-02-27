# H200 Ground Truth — Analysis Tables

Generated: analyze_h200_ground_truth.py


## T1: Batch TPOT (ms)

Lower is better. **Bold** = winner per column.


### i=128 (TP4 excluded)

| Config | b=1 | b=8 | b=16 | b=32 | b=64 | Δ(1→64) |
|--------|------:|------:|------:|------:|------:|---------:|
| TP2 H200 NVLink | 5.83 | **6.46** | **6.85** | **7.79** | 9.94 | 1.7x |
| TP8 H200 NVLink | **4.58** | 6.60 | 7.06 | 8.62 | **9.28** | 2.0x |
| TP8+EP H200 NVLink | 5.01 | 10.60 | 7.40 | 8.62 | 10.36 | 2.1x |
| **Winner** | TP8 | TP2 | TP2 | TP2 | TP8 | |

### i=512

| Config | b=1 | b=8 | b=16 | b=32 | b=64 | Δ(1→64) |
|--------|------:|------:|------:|------:|------:|---------:|
| TP2 H200 NVLink | 5.93 | 6.37 | 7.03 | **7.86** | 10.30 | 1.7x |
| TP4 H200 NVLink | 5.39 | **6.00** | **6.08** | 8.79 | 15.66 | 2.9x |
| TP8 H200 NVLink | **4.32** | 6.14 | 6.65 | 8.75 | **9.43** | 2.2x |
| TP8+EP H200 NVLink | 4.74 | 7.43 | 7.80 | 11.66 | 12.24 | 2.6x |
| **Winner** | TP8 | TP4 | TP4 | TP2 | TP8 | |

### i=2048

| Config | b=1 | b=8 | b=16 | b=32 | b=64 | Δ(1→64) |
|--------|------:|------:|------:|------:|------:|---------:|
| TP2 H200 NVLink | 6.32 | 6.95 | 7.66 | 8.84 | 10.05 | 1.6x |
| TP4 H200 NVLink | 5.39 | **6.02** | **6.38** | **7.53** | **9.12** | 1.7x |
| TP8 H200 NVLink | **4.32** | 6.66 | 7.24 | 9.09 | 10.34 | 2.4x |
| TP8+EP H200 NVLink | 4.76 | 8.34 | 10.09 | 13.64 | 20.83 | 4.4x |
| **Winner** | TP8 | TP4 | TP4 | TP4 | TP4 | |

- **TP4** wins 6/15 columns.
- Steepest degradation: TP8+EP i=2048 (4.4x from b=1→b=64).
- i=128 b=8: TP8+EP is 64% worse (6.5 vs 10.6)
- i=512 b=64: TP4 is 66% worse (9.4 vs 15.7)
- i=2048 b=16: TP8+EP is 58% worse (6.4 vs 10.1)

## T2: Batch TPS

Higher is better. **Bold** = winner per column.


### i=128 (TP4 excluded)

| Config | b=1 | b=8 | b=16 | b=32 | b=64 | Δ(1→64) |
|--------|------:|------:|------:|------:|------:|---------:|
| TP2 H200 NVLink | 106 | 734 | 1380 | 2310 | 3782 | 35.6x |
| TP8 H200 NVLink | **189** | **1133** | **2136** | 3522 | **6497** | 34.4x |
| TP8+EP H200 NVLink | 174 | 725 | 2038 | **3522** | 5889 | 33.8x |
| **Winner** | TP8 | TP8 | TP8 | TP8+EP | TP8 | |

### i=512

| Config | b=1 | b=8 | b=16 | b=32 | b=64 | Δ(1→64) |
|--------|------:|------:|------:|------:|------:|---------:|
| TP2 H200 NVLink | 106 | 736 | 1373 | 2245 | 3475 | 32.8x |
| TP4 H200 NVLink | 9 | 789 | 1489 | 2253 | 2677 | 284.8x |
| TP8 H200 NVLink | **185** | **1219** | **2256** | **3470** | **6428** | 34.7x |
| TP8+EP H200 NVLink | 160 | 1016 | 1940 | 2625 | 4966 | 31.1x |
| **Winner** | TP8 | TP8 | TP8 | TP8 | TP8 | |

### i=2048

| Config | b=1 | b=8 | b=16 | b=32 | b=64 | Δ(1→64) |
|--------|------:|------:|------:|------:|------:|---------:|
| TP2 H200 NVLink | 105 | 638 | 1130 | 1896 | 3375 | 32.0x |
| TP4 H200 NVLink | 74 | 742 | 1246 | 2150 | 3500 | 47.3x |
| TP8 H200 NVLink | **185** | **1124** | **2081** | **3327** | **5877** | 31.8x |
| TP8+EP H200 NVLink | 159 | 909 | 1513 | 2244 | 2945 | 18.5x |
| **Winner** | TP8 | TP8 | TP8 | TP8 | TP8 | |

- **TP8** wins 14/15 columns.
- i=128 b=1: TP2 is 78% worse (106.1 vs 188.9)
- i=128 b=8: TP8+EP is 56% worse (724.8 vs 1133.3)
- i=128 b=16: TP2 is 55% worse (1379.6 vs 2136.2)

## T3: Batch TTFT (ms) — 8GPU Only

Lower is better. **Bold** = winner. TTFT constant across batch sizes.

> 4GPU TTFT excluded (scheduling artifacts: 400–12000ms)

| Config | i=128 | i=512 | i=2048 | EP overhead |
|--------|------:|------:|------:|------------|
| TP8 H200 NVLink | **62.7** | **91.0** | **96.1** | (baseline) |
| TP8+EP H200 NVLink | 65.3 | 148.1 | 150.7 | +38.1ms avg |

- **TP8** wins all input lengths (no EP routing overhead).
- i=128: EP adds +2.6ms (4% overhead).
- i=512: EP adds +57.2ms (63% overhead).
- i=2048: EP adds +54.6ms (57% overhead).

## T4: Agentic TPOT (ms)

Lower is better. **Bold** = winner.

| Config | c=1 | c=8 | c=32 | c=64 | c=128 | Δ(min→max) |
|--------|------:|------:|------:|------:|------:|----------:|
| TP2 H200 NVLink | 7.69 | 10.58 | 22.02 | 30.03 | — | 3.9x |
| TP4 H200 NVLink | 6.74 | 8.50 | 20.20 | 21.93 | — | 3.3x |
| TP8 H200 NVLink | **4.94** | **8.43** | **15.14** | **16.81** | **16.45** | 3.4x |
| TP8+EP H200 NVLink | 5.39 | 8.52 | 15.70 | 17.91 | 19.56 | 3.6x |
| TP4 H100 PCIe | 7.26 | 12.39 | 24.71 | 25.96 | — | 3.6x |
| **Winner** | TP8 | TP8 | TP8 | TP8 | TP8 | |

- **TP8** wins 5/5 columns.
- Max degradation: TP2 (3.9x).

## T5: Agentic TPS

Higher is better. **Bold** = winner.

| Config | c=1 | c=8 | c=32 | c=64 | c=128 | Peak |
|--------|------:|------:|------:|------:|------:|-----:|
| TP2 H200 NVLink | 118 | 585 | 902 | 1330 | — | 1330 |
| TP4 H200 NVLink | 128 | 707 | 1179 | 1589 | — | 1589 |
| TP8 H200 NVLink | 175 | **841** | 1444 | **1626** | **1648** | 1648 |
| TP8+EP H200 NVLink | **176** | 817 | **1495** | 1622 | 1444 | 1622 |
| TP4 H100 PCIe | 130 | 568 | 882 | 876 | — | 882 |
| **Winner** | TP8+EP | TP8 | TP8+EP | TP8 | TP8 | |

- **TP8** wins 3/5 columns.

## T6: Agentic TTFT (ms)

Lower is better. **Bold** = winner.

> 4GPU H200 TTFT excluded (scheduling artifacts)

| Config | c=1 | c=8 | c=32 | c=64 | c=128 | Δ(min→max) |
|--------|------:|------:|------:|------:|------:|----------:|
| TP8 H200 NVLink | 158.9 | 114.5 | 191.9 | 171.8 | 180.1 | 1.7x |
| TP8+EP H200 NVLink | **82.9** | **96.5** | **146.7** | **168.9** | **179.5** | 2.2x |
| **Winner** | TP8+EP | TP8+EP | TP8+EP | TP8+EP | TP8+EP | |

- **TP8+EP** wins 5/5 columns.
- Max degradation: TP8+EP (2.2x).

## T7: Platform — TP4 H200 NVLink vs TP4 H100 PCIe

| Metric | c=1 | c=8 | c=32 | c=64 |
|--------|------:|------:|------:|------:|
| H200 NVLink TPOT (ms) | 6.7 | 8.5 | 20.2 | 21.9 |
| H100 PCIe TPOT (ms) | 7.3 | 12.4 | 24.7 | 26.0 |
| **Δ H100−H200 (ms)** | +0.5 | +3.9 | +4.5 | +4.0 |

| Metric | c=1 | c=8 | c=32 | c=64 |
|--------|------:|------:|------:|------:|
| H200 NVLink TPS | 128 | 707 | 1179 | 1589 |
| H100 PCIe TPS | 130 | 568 | 882 | 876 |
| **Δ H200−H100** | +-2 | +138 | +297 | +713 |

- c=1: H200 is **1.1x** faster TPOT, +-1% TPS.
- c=8: H200 is **1.5x** faster TPOT, +24% TPS.
- c=32: H200 is **1.2x** faster TPOT, +34% TPS.
- c=64: H200 is **1.2x** faster TPOT, +81% TPS.

## T8: DP Batch TPOT (ms)

Lower is better. **Bold** = winner per column.

> DP configs only. TP2-DP4-EP i=128 b=8 excluded (anomaly).


### i=128

| Config | b=1 | b=8 | b=16 | b=32 | b=64 | Δ(1→64) |
|--------|------:|------:|------:|------:|------:|---------:|
| TP2-DP4-EP H200 NVLink | 8.15 | — | **10.15** | **10.50** | **11.72** | 1.4x |
| TP4-DP2 H200 NVLink | 8.01 | 9.74 | 10.19 | 11.58 | 13.04 | 1.6x |
| TP4-DP2-EP H200 NVLink | **7.69** | **9.53** | 10.18 | 11.57 | 12.97 | 1.7x |
| **Winner** | TP4-DP2-EP | TP4-DP2-EP | TP2-DP4-EP | TP2-DP4-EP | TP2-DP4-EP | |

### i=512

| Config | b=1 | b=8 | b=16 | b=32 | b=64 | Δ(1→64) |
|--------|------:|------:|------:|------:|------:|---------:|
| TP2-DP4-EP H200 NVLink | **7.95** | **10.05** | **10.43** | **11.44** | **13.39** | 1.7x |
| TP4-DP2 H200 NVLink | 8.37 | 10.42 | 11.33 | 16.46 | 18.67 | 2.2x |
| TP4-DP2-EP H200 NVLink | 8.44 | 10.40 | 11.32 | 17.24 | 19.07 | 2.3x |
| **Winner** | TP2-DP4-EP | TP2-DP4-EP | TP2-DP4-EP | TP2-DP4-EP | TP2-DP4-EP | |

### i=2048

| Config | b=1 | b=8 | b=16 | b=32 | b=64 | Δ(1→64) |
|--------|------:|------:|------:|------:|------:|---------:|
| TP2-DP4-EP H200 NVLink | 7.91 | **10.06** | **11.27** | **13.90** | **19.23** | 2.4x |
| TP4-DP2 H200 NVLink | **7.17** | 11.69 | 14.88 | 21.74 | 31.85 | 4.4x |
| TP4-DP2-EP H200 NVLink | 8.49 | 11.72 | 14.79 | 20.28 | 31.44 | 3.7x |
| **Winner** | TP4-DP2 | TP2-DP4-EP | TP2-DP4-EP | TP2-DP4-EP | TP2-DP4-EP | |

- **TP2-DP4-EP** wins 12/15 columns.
- Steepest degradation: TP4-DP2 i=2048 (4.4x from b=1→b=64).
- i=512 b=32: TP4-DP2-EP is 51% worse (11.4 vs 17.2)
- i=2048 b=32: TP4-DP2 is 56% worse (13.9 vs 21.7)
- i=2048 b=64: TP4-DP2 is 66% worse (19.2 vs 31.9)

## T9: TP vs DP — TPOT Ratio

Ratio = DP TPOT / TP8 TPOT. >1 means DP is slower per-request.


### i=128

| Comparison | b=1 | b=8 | b=16 | b=32 | b=64 |
|------------|------:|------:|------:|------:|------:|
| TP4-DP2/TP8 | 1.75x | 1.48x | 1.44x | 1.34x | 1.41x |
| TP2-DP4-EP/TP8 | 1.78x | — | 1.44x | 1.22x | 1.26x |

### i=512

| Comparison | b=1 | b=8 | b=16 | b=32 | b=64 |
|------------|------:|------:|------:|------:|------:|
| TP4-DP2/TP8 | 1.94x | 1.70x | 1.70x | 1.88x | 1.98x |
| TP2-DP4-EP/TP8 | 1.84x | 1.64x | 1.57x | 1.31x | 1.42x |

### i=2048

| Comparison | b=1 | b=8 | b=16 | b=32 | b=64 |
|------------|------:|------:|------:|------:|------:|
| TP4-DP2/TP8 | 1.66x | 1.76x | 2.06x | 2.39x | 3.08x |
| TP2-DP4-EP/TP8 | 1.83x | 1.51x | 1.56x | 1.53x | 1.86x |

- Ratio >1: DP is slower per-request (but total throughput scales with DP factor).
- Ratio ~1: DP matches TP8 per-request TPOT.
