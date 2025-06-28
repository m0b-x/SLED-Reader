<p align="center">
  <img src="https://github.com/user-attachments/assets/80b632f6-e243-4288-b8a7-29f72c9634ff" width="800"/>
</p>

# SLED Reader – Accelerometer Filtering & Visualization Toolkit

**SLED Reader** is a Python-based desktop application for visualizing, filtering, and analyzing accelerometer data. Designed around the **Step Length Estimation Dataset (SLED)** format, it enables intuitive signal processing experimentation for researchers and students working in inertial sensing, gait analysis, or mobile health.

## 🔧 Features

- Load and preview **SLED-format JSON files**
- Apply multiple real-time **filtering pipelines**:
  - Low-pass, high-pass, and band-pass filters (Butterworth)
  - **FIR filters**
  - **Kalman filters**, **Net Magnitude** filtering, and **Moving Average**
  - Min‑Max scaling and signal normalization
- Visualize:
  - Raw acceleration components (X, Y, Z)
  - Magnitude and filtered magnitude
  - Peaks, valleys, and **zero-crossing points**
- Customizable plots with interactive label editing

## Screenshots

![normal_copy1](https://github.com/user-attachments/assets/95fb6eda-e70d-47ab-bf31-9107cd51c32e)


## 🧪 Educational Use

This app is ideal for:
- Analyzing and comparing signal preprocessing methods
- Preparing data for step detection or activity recognition
- Classroom demos and lab projects in any accelerometer processing scenarios

## ⚙️ Built With

- **Python 3** + **Tkinter GUI**
- Uses:
  - `matplotlib` for plotting
  - `scipy.signal` for filtering and peak detection
  - Native JSON + CSV support for I/O

### Modular Structure:
- `compute_trace()` – applies all filters and builds labeled traces  
- `plot_files()` – plots single or multi-file overlays  
- `compute_stats_from_magnitude()` – computes peak/valley and RMS features  
- `open_legend_editor()` – dynamic plot relabeling  

## 📊 Output & Analysis

- Signal stats for each file:
  - Mean/STD of peaks & valleys  
  - Peak/valley count  
  - RMS & zero-crossings  
- Export-ready visualizations and summaries for academic writing or presentations

## 📁 Dataset Compatibility

Compatible with the **Step Length Estimation Dataset (SLED)** format, including:
- JSON fields: `linear_acceleration`, `height`, `gender`, `leg_length`, `smartphone_position`
- Automatic metadata extraction from filenames and file content
  
🔗 Official SLED Dataset Repository: [github.com/repositoryadmin/SLERepository](https://github.com/repositoryadmin/SLERepository/tree/master)

## 🚀 Getting Started

1. Clone the repo  
2. Install dependencies: `pip install matplotlib numpy scipy`  
3. Run with `python main.py`  
4. Load SLED JSON files and start visualizing!

## 📬 Author & Contact

This program was developed by **Alexandru Zamfir**.  
If you have questions, suggestions, or want to collaborate, feel free to reach out:

📧 [alexandru.zamfir@proton.me](mailto:alexandru.zamfir@proton.me)

## 📝 Dataset Acknowledgment

This application builds upon the [**SLED benchmark datasets**](https://github.com/repositoryadmin/SLERepository/tree/master), created and maintained by [Melanija Vezočnik](mailto:contact.sle.repository@gmail.com) and colleagues at the **University of Ljubljana**

These datasets enable performance evaluation of inertial sensor-based step length estimation models, offering over 30 hours of labeled gait data across various walking speeds, positions, and conditions.

Researchers are encouraged to explore and contribute to this open benchmarking effort.

> **Contact:**  
> Melanija Vezočnik – [contact.sle.repository@gmail.com](mailto:contact.sle.repository@gmail.com)  
> Laboratory for Integration of Information Systems  
> Faculty of Computer and Information Science  
> University of Ljubljana, Slovenia

---

> Developed as part of a research toolkit for evaluating signal preprocessing methods on inertial datasets such as SLED.
