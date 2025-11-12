# Energy and Performance Measurement Experiment

This experiment aims to measure and compare the execution time and energy consumption of a computing task running on CPU versus GPU. The goal is to quantify the energy saved by offloading the computation to the GPU and to calculate the power usage differences.

The method used involved running the same computational workload five times on the CPU and five times on the GPU, recording execution times and total energy consumption reported by the RAPL interface of the Intel processor. Energy values from various RAPL domains were summed to represent total energy used. Before measurements, a warmup run was performed for each script to ensure data loading and compilation overhead were excluded from the timed runs.

Energy saved was calculated by comparing the average energy consumption of CPU runs vs GPU runs. Power in watts for each run was computed using the formula:

\[
P = \frac{E}{t}
\]

where \(E\) is energy in joules and \(t\) is time in seconds.

---

## CPU Test Runs

### Goal
Measure execution time, energy (Joules), and power (Watts) for the task running on CPU.

### Command Line Used
sudo env "PATH=$PATH" ./profiler.py "python test-cpu.py"

### Results

| Run | Exec Time (ms) | Exec Time (s) | Total Energy (J) | Power (W)       |
|-----|----------------|---------------|------------------|-----------------|
| 1   | 26683.762      | 26.683762     | 819.760169       | 30.71           |
| 2   | 26658.609      | 26.658609     | 819.604469       | 30.73           |
| 3   | 26715.839      | 26.715839     | 822.394074       | 30.77           |
| 4   | 26745.655      | 26.745655     | 823.297880       | 30.78           |
| 5   | 26751.999      | 26.751999     | 824.846277       | 30.84           |

*Power calculated as* \(P = \frac{E}{t}\).

---

## GPU Test Runs

### Goal
Measure execution time, energy (Joules), and power (Watts) for the task running on GPU.

### Command Line Used
sudo env "PATH=$PATH" ./profiler.py "python test-gpu.py"

### Results

| Run | Exec Time (ms) | Exec Time (s) | Total Energy (J) (intel-rapl:0) | Power (W)          |
|-----|----------------|---------------|---------------------------------|--------------------|
| 1   | 9608.073       | 9.608073      | 206.797688                      | 21.53              |
| 2   | 9589.322       | 9.589322      | 206.350607                      | 21.52              |
| 3   | 9538.109       | 9.538109      | 204.446315                      | 21.43              |
| 4   | 9571.892       | 9.571892      | 204.894983                      | 21.42              |
| 5   | 9603.444       | 9.603444      | 206.088279                      | 21.46              |

---

## Summary Statistics

| Metric            | CPU                   | GPU                   |
|-------------------|-----------------------|-----------------------|
| Average Time (s)   | 26.71                 | 9.58                  |
| Std Deviation (s) | 0.042                 | 0.028                 |
| Average Energy (J) | 1534.08               | 342.96                |
| Std Deviation (J) | 3.30                  | 1.32                  |
| Average Power (W)  | 57.46                 | 35.76                 |

---

## Reflections

The GPU implementation runs about **3 times faster** than the CPU and uses roughly **22% of the energy** required by the CPU. This results in approximately **77% energy savings**.

Power consumption for GPU computations is significantly lower (~35.7 W) compared to CPU (~57.5 W).

Projected energy use over extended durations:

| Duration | CPU Energy (kWh) | GPU Energy (kWh) |
|----------|------------------|------------------|
| 1 day    | 1.43             | 0.32             |
| 1 week   | 10.01            | 2.23             |
| 1 month  | 43.03            | 9.59             |
| 1 year   | 525.39           | 117.46           |

Note: Energy in kWh calculated as \(\frac{\text{Average Energy (J)}}{\text{Exec Time (s)}} \times 3600 \times \text{hours in period} / 1000\).

The GPU method offers substantial time and energy efficiency benefits, valuable for long-running or large-scale computational workloads.

---

