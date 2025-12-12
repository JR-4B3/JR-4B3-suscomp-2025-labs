# Lab Report: Measuring Cloud Emissions

## 1. Introduction
The goal of this lab exercise is to measure energy usage and understand hidden emissions when running software in a cloud environment such as Google Colab. Since virtual machines (VMs) do not provide access to low-level hardware counters (like Intel RAPL), we rely on software-level measurements to approximate energy usage for CPU, GPU, and networking. These measurements are estimates rather than exact ground-truth values.

The key challenges noted for measurements are:
* Avoiding data downloads/uploads inside the measurement window.
* Accounting for the small overhead of the utilization and power logging Python thread.
* Acknowledge that DRAM power, NVMe/SSD power, and CPU package components are not directly measurable.
* Repeating tests (3 times) to report mean and standard deviation.

## 2. GPU Cold Start Experiment
We measured the time required for the cloud GPU to initialize its environment (cold start) versus subsequent warm execution.

* **Initialization time (first run):** 0.05127 s
* **Second run time:** 0.00012 s

**Observation:**
The first run is much slower because the GPU needs to be "woken up," and the CUDA context and related resources must be initialized, which introduces cold-start latency. Subsequent calls reuse the existing context, resulting in a near-instantaneous second run.

---

## 3. CPU vs. GPU Performance: Matrix Multiplication

### 3.1 Raw Timings & Performance
We compared the performance of an estimated **Intel Xeon Server CPU (85–105 W TDP)** against an assumed **NVIDIA Tesla T4 GPU (70 W)** on a large matrix multiplication task over three runs.

| Run | CPU Time (s) | GPU Time (s) |
| :---: | :---: | :---: |
| 1 | 33.69772 | 0.14616 |
| 2 | 33.81726 | 0.00040 |
| 3 | 34.34183 | 0.00038 |
| **Average** | **33.95 s** | **0.049 s** |

**Speedup Calculation:**
The GPU speedup is calculated as $\text{Speedup} \approx \frac{33.95 \text{ s}}{0.049 \text{ s}} \approx 690\times$.

The GPU is about **690× faster** than the CPU for this task.

### 3.2 Energy Estimation (Rough)
We estimate energy ($E = P \cdot t$) using typical power assumptions:
* CPU Power ($P_{\text{CPU}}$): $100 \text{ W}$
* GPU Power ($P_{\text{GPU}}$): $70 \text{ W}$

**CPU Energy:**
$$
E_{\text{CPU}} \approx 100 \text{ W} \cdot 33.95 \text{ s} = 3395 \text{ W} \cdot \text{s}
$$
$$
E_{\text{CPU, kWh}} \approx 3395 / 3,600,000 \approx 9.4 \times 10^{-4} \text{ kWh}
$$

**GPU Energy:**
$$
E_{\text{GPU}} \approx 70 \text{ W} \cdot 0.049 \text{ s} \approx 3.43 \text{ W} \cdot \text{s}
$$
$$
E_{\text{GPU, kWh}} \approx 3.43 / 3,600,000 \approx 9.5 \times 10^{-7} \text{ kWh}
$$

### 3.3 Observation
Even though the GPU's instantaneous power draw is of similar magnitude to the CPU's, the GPU completes the computation much faster. As a result, the **total energy for this matrix multiplication is about three orders of magnitude lower on the GPU**, while delivering a **690× speedup**.

---

## 4. Cloud VM Geolocation and Carbon Intensity

### 4.1 VM Location
Using IP geolocation, the Google Colab VM was determined to be running in a Google data center in **The Dalles, Oregon, USA**.

| Field | Value |
| :--- | :--- |
| IP | 35.230.77.168 |
| City | The Dalles |
| Region | Oregon |
| Country | US |

### 4.2 Electricity Mix and Carbon Intensity
The VM location corresponds to the PacifiCorp West ($US-NW-PACW$) zone.

**Observed Electricity Mix:**
* Hydro: **24.96 %**
* Wind: **18.60 %**
* Gas: **32.22 %**
* Unknown/Other: **2.81 %**

The direct carbon intensity for this zone at the time of query was **196 gCO₂e/kWh**.

### 4.3 Emissions Calculation (Matrix Multiplication)
Using the carbon intensity of $196 \text{ gCO}_2\text{e/kWh}$:

* **CPU Emissions:**
    $$
    \text{Emissions} \approx 9.4 \times 10^{-4} \text{ kWh} \times 196 \text{ gCO}_2\text{e/kWh} \approx \mathbf{0.184 \text{ g CO}_2\text{e}}
    $$

* **GPU Emissions:**
    $$
    \text{Emissions} \approx 9.5 \times 10^{-7} \text{ kWh} \times 196 \text{ gCO}_2\text{e/kWh} \approx \mathbf{1.9 \times 10^{-4} \text{ g CO}_2\text{e}}
    $$

**Conclusion:** The GPU is both much faster and dramatically lower-emission for this computation due to its superior energy efficiency on parallel workloads.

---

## 5. Data Transfer Analysis

### 5.1 Dataset Transfer Scenarios (MNIST 10.95 MB)
We compared the time taken to transfer a file using four different paths.

| Path | File Size | Time (s) | Transfer Rate / Bottleneck |
| :--- | :--- | :--- | :--- |
| Cloud Storage $\to$ Local Machine | 10.95 MB | $\approx 0.50 \text{ s}$ | $\approx 22.02 \text{ MB/s}$ (Local download limit) |
| **Local Machine $\to$ Colab VM** | 10.95 MB | **79.11 s** | Severely limited by home uplink bandwidth. |
| Colab VM $\to$ Local Machine | 10.95 MB | Fast | Limited by local download bandwidth. |
| **Cloud Storage $\to$ Colab VM** | 10.96 MB | **$\approx 0.03 \text{ s}$** | $\approx 365 \text{ MB/s}$ (Internal cloud network) |

**Key Finding:** Downloading data directly from Google Cloud Storage to the Colab VM is orders of magnitude faster than staging it through the local computer.

### 5.2 Network Proximity (Colab VM to Cloud Storage)
* **Ping RTT:** $\text{min/avg/max} \approx 0.274 / 0.357 / 0.455 \text{ ms}$. This sub-millisecond average round-trip time strongly suggests the Colab VM and the storage endpoint are in the same data center campus or extremely close metropolitan region.
* **Traceroute:** The path from the Colab VM to `storage.googleapis.com` involved only a handful of internal Google routers, reaching the destination in well under 2 ms RTT. In contrast, the local machine had $\approx 22$ hops and $\approx 34 \text{ ms}$ RTT.

**Conclusion:** Data transfer between the Colab VM and Google Cloud Storage is effectively "local" in network terms, explaining the very high throughput observed.

---

## 6. ML Training Energy Consumption: YearPredictionMSD Dataset

### 6.1 Dataset Loading
The YearPredictionMSD dataset ($\approx 201.24 \text{ MB}$) was downloaded directly to the Colab VM in $\approx 2.3 \text{ s}$. However, loading, parsing, and converting the compressed CSV file into NumPy arrays took approximately **11.20 s**. This shows that the bottleneck for data preparation was CPU/Disk I/O processing inside the VM, not network transfer speed.

### 6.2 ML Training Results (Linear Regression)
We trained a model using both CPU and GPU (PyTorch), measuring time and energy use.

| Metric (Average of 3 Runs) | CPU Training | GPU Training |
| :--- | :--- | :--- |
| **Training Time** | $\approx 42.96 \text{ s}$ | $\approx 45.95 \text{ s}$ |
| **CPU Energy** | $5.37 \times 10^{-4} \text{ kWh}$ | $5.64 \times 10^{-4} \text{ kWh}$ |
| **GPU Energy** | $3.35 \times 10^{-4} \text{ kWh}$ (Idle) | $3.59 \times 10^{-4} \text{ kWh}$ (Active) |
| **Total Energy** | $\approx 8.72 \times 10^{-4} \text{ kWh}$ | $\approx 9.23 \times 10^{-4} \text{ kWh}$ |
| **Emissions** ($196 \text{ gCO}_2\text{e/kWh}$) | $\approx 0.171 \text{ g CO}_2\text{e}$ | $\approx 0.181 \text{ g CO}_2\text{e}$ |

*(Note: The raw snippets only provided emissions based on the active component. Total energy considers both CPU and GPU consumption during the run.)*

### 6.3 Final Conclusion

1.  **Efficiency vs. Scale:** The GPU demonstrated orders of magnitude greater efficiency (690× speedup, $\sim 1000\times$ less energy) on the heavy, parallelizable **matrix multiplication** task. However, for the lighter **linear regression** training, the GPU was not significantly faster, and the difference in energy/emissions was small, showing that model complexity and CPU-GPU data transfer overheads can negate the GPU's advantage on simpler workloads.
2.  **Location Matters:** The choice of the **The Dalles, Oregon** region provides a relatively low carbon intensity ($196 \text{ gCO}_2\text{e/kWh}$) due to significant hydropower/wind in the grid mix.
3.  **Data Movement:** Network latency analysis confirms that storing datasets in cloud storage near the compute VM is crucial, as the bottleneck for cloud workloads is often the slow uplink from the user's local machine ($79 \text{ s}$ upload) rather than the internal cloud network ($0.03 \text{ s}$ download).