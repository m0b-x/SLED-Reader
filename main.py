import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
import re

matplotlib.use("TkAgg")


# ----------------- Helper functions -----------------

def compute_magnitude(x, y, z):
    xyz = np.vstack((x, y, z)).astype(float)
    return np.linalg.norm(xyz, axis=0)


def butter_filter(data, cutoff, fs, order, kind):
    nyq = 0.5 * fs
    if isinstance(cutoff, (list, tuple)):
        normal_cutoff = [c / nyq for c in cutoff]
    else:
        normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype=kind, analog=False, output='sos')
    return sosfiltfilt(sos, data)


def minmax_scale(data, target_min=0.0, target_max=1.0):
    data = np.asarray(data)
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return np.full_like(data, target_min)
    return (data - min_val) / (max_val - min_val) * (target_max - target_min) + target_min


def moving_average_filter(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def net_magnitude_filter(data, window_size):
    data = np.asarray(data)
    net_mag = np.zeros_like(data)
    for i in range(len(data)):
        if i < window_size:
            net_mag[i] = data[i]
        else:
            net_mag[i] = data[i] - np.mean(data[i - window_size:i])
    return net_mag


def kalman_filter(data):
    n = len(data)
    x_est = np.zeros(n)
    p = np.zeros(n)
    x_est[0] = data[0]
    p[0] = 1.0
    Q = 0.01  # process variance
    R = 1.0  # measurement variance
    for k in range(1, n):
        x_pred = x_est[k - 1]
        p_pred = p[k - 1] + Q
        K = p_pred / (p_pred + R)
        x_est[k] = x_pred + K * (data[k] - x_pred)
        p[k] = (1 - K) * p_pred
    return x_est


def create_entry_with_hz(parent, default_value=""):
    frame = tk.Frame(parent, bg="white")
    entry = tk.Entry(frame, width=6, justify="center")
    entry.insert(0, default_value)
    entry.pack(side="left")
    tk.Label(frame, text="Hz,", bg="white").pack(side="left", padx=(2, 0))
    return frame, entry


# ----------------- Core app logic -----------------

loaded_files = {}
meta = {}
FS = 100.0


def load_file():
    paths = filedialog.askopenfilenames(title="Select JSON files", filetypes=[("JSON files", "*.json")])
    if not paths:
        return
    for path in paths:
        if path in loaded_files:
            continue
        try:
            with open(path, "r") as f:
                d = json.load(f)
            height = d.get("height", "?")
            gender = d.get("gender", "?")
            leg_length = d.get("leg_length", "?")
            position = d.get("smartphone_position", "?")
            filename = path.split("/")[-1].split(".")[0]
            match = re.search(r"(fast|slow|normal|preferred)", filename.lower())
            speed = match.group(1).capitalize() if match else "?"
            meta[path] = {"height": height, "gender": gender, "speed": speed, "leg_length": leg_length,
                          "position": position}
            height_str = f"{float(height):.2f} m" if isinstance(height, (int, float, str)) and str(height).replace('.',
                                                                                                                   '',
                                                                                                                   1).isdigit() else str(
                height)
            leg_str = f"{float(leg_length):.2f} m" if isinstance(leg_length, (int, float, str)) and str(
                leg_length).replace('.', '', 1).isdigit() else str(leg_length)
            display_name = path.split("/")[-1]
            loaded_files[path] = display_name
            tree.insert("", "end", iid=path, values=(display_name, gender, height_str, leg_str, speed, position))
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file:\n{path}\n\n{e}")



def clear_selected():
    selected_items = tree.selection()
    for item in selected_items:
        tree.delete(item)
        loaded_files.pop(item, None)
        meta.pop(item, None)

def clear_all():
    loaded_files.clear()
    meta.clear()
    for item in tree.get_children():
        tree.delete(item)


def validate_indices():
    if var_read_all.get():
        return 0, None
    try:
        start = int(entry_start.get())
        end = int(entry_end.get())
        if start < 0 or end <= start:
            raise ValueError
        return start, end
    except ValueError:
        messagebox.showerror("Invalid range", "Please enter valid integers for start and end (start < end).")
        return None, None


def compute_trace(path, start, end):
    with open(path, "r") as f:
        d = json.load(f)
    x = d["linear_acceleration"]["x"][start:end]
    y = d["linear_acceleration"]["y"][start:end]
    z = d["linear_acceleration"]["z"][start:end]
    if end is None:
        end = len(x)
    base = path.split("/")[-1].split(".")[0]
    gender = meta.get(path, {}).get("gender", "?").capitalize()
    height = meta.get(path, {}).get("height", "?")
    speed = meta.get(path, {}).get("speed", "?").capitalize()
    position = meta.get(path, {}).get("position", "?").capitalize()
    try:
        height_str = f"{float(height):.2f}m"
    except (ValueError, TypeError):
        height_str = height
    tag_parts = []
    if var_show_name.get():
        tag_parts.append(base)
    if var_show_stats.get():
        tag_parts.append(f"{gender}, {height_str}, {speed} Walking Speed, {position} Position")
    tag = " | ".join(tag_parts)
    mag = compute_magnitude(x, y, z)

    def lab(suffix):
        suffix_map = {
            "raw": "Raw",
            "LP": "Low‑passed",
            "HP": "High‑passed",
            "Band-pass": "Band‑passed",
            "Kalman": "Kalman",
            "NetMag": "Net Magnitude",
            "MA": "Moving Average",
            "Filtered": "Filtered"
        }
        readable = suffix_map.get(suffix, suffix.capitalize())
        return f"{tag} ({readable})" if tag else readable

    traces = []

    if var_show_components.get():
        traces.append((f"{tag} (X)", x))
        traces.append((f"{tag} (Y)", y))
        traces.append((f"{tag} (Z)", z))

    #####################################if not var_hide_magnitude.get():
    filtered = mag.copy()

    suffixes = []

    # Filtering: LP, HP, Band-pass
    lp_enabled = var_lowpass.get()
    hp_enabled = var_highpass.get()
    if lp_enabled and hp_enabled:
        lp_cut = float(entry_lp_cutoff.get())
        hp_cut = float(entry_hp_cutoff.get())
        order = int(entry_lp_order.get())

        if var_split_band.get():
            lp_filtered = butter_filter(mag, lp_cut, FS, order, 'low')
            hp_filtered = butter_filter(mag, hp_cut, FS, order, 'high')
            if var_show_raw.get():
                traces.append((lab("raw"), mag))
            traces.append((lab("LP"), lp_filtered))
            traces.append((lab("HP"), hp_filtered))
            return traces

        else:
            filtered = butter_filter(filtered, [hp_cut, lp_cut], FS, order, 'band')
            suffixes.append("Band-pass")
    elif lp_enabled:
        lp_cut = float(entry_lp_cutoff.get())
        order = int(entry_lp_order.get())
        filtered = butter_filter(filtered, lp_cut, FS, order, 'low')
        suffixes.append("LP")
    elif hp_enabled:
        hp_cut = float(entry_hp_cutoff.get())
        order = int(entry_hp_order.get())
        filtered = butter_filter(filtered, hp_cut, FS, order, 'high')
        suffixes.append("HP")

    # Kalman
    if var_kalman.get():
        filtered = kalman_filter(filtered)
        suffixes.append("Kalman")

    # Net Magnitude
    if var_net_mag.get():
        try:
            window_size = int(entry_net_window.get())
            if window_size > 1:
                filtered = net_magnitude_filter(filtered, window_size)
                suffixes.append(f"NetMag-{window_size}")
        except ValueError:
            messagebox.showerror("Invalid input", "Net magnitude window must be an integer > 1.")

    # MinMax Scaling
    if var_minmax.get():
        try:
            target_min = float(entry_minmax_min.get())
            target_max = float(entry_minmax_max.get())
            filtered = minmax_scale(filtered, target_min, target_max)
            suffixes.append(f"MinMax({target_min}-{target_max})")
        except ValueError:
            messagebox.showerror("Invalid input", "MinMax values must be numeric.")

    # Moving Average
    if var_moving_avg.get():
        try:
            window_size = int(entry_ma_window.get())
            if window_size > 1:
                filtered = moving_average_filter(filtered, window_size)
                suffixes.append(f"MA-{window_size}")
        except ValueError:
            messagebox.showerror("Invalid input", "Moving average window must be an integer > 1.")

    label_suffix = " + ".join(suffixes) if suffixes else "Filtered"

    if var_show_raw.get():
        traces.append((lab("raw"), mag))
        if suffixes:
            traces.append((lab(label_suffix), filtered))
    else:

        if suffixes:
            traces = [(lab(label_suffix), filtered)]
            if var_zero_crossing.get():
                filtered = np.asarray(filtered)
                signs = np.sign(filtered)
                crossings = np.where(np.diff(signs))[0]
                x_cross = []
                y_cross = []
                for idx in crossings:
                    x0, x1 = idx, idx + 1
                    y0, y1 = filtered[x0], filtered[x1]
                    if y1 - y0 != 0:
                        x_zero = x0 - y0 * (x1 - x0) / (y1 - y0)
                        x_cross.append(x_zero)
                        y_cross.append(0)
                traces.append(("__zero_cross__", (x_cross, y_cross)))
        else:
            traces = [(lab("raw"), mag)]

    ##############################################
    return traces


def open_legend_editor(current_labels):
    editor = tk.Toplevel()
    editor.title("Edit Plot Labels")

    tk.Label(editor, text="Plot Title:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
    entry_plot_title = tk.Entry(editor, width=60)
    entry_plot_title.insert(0, plt.gca().get_title())
    entry_plot_title.grid(row=0, column=1, padx=5, pady=5)

    entries = []
    for i, label in enumerate(current_labels):
        tk.Label(editor, text=f"Label {i + 1}:").grid(row=i + 1, column=0, padx=5, pady=3, sticky="e")
        entry = tk.Entry(editor, width=60)
        entry.insert(0, label)
        entry.grid(row=i + 1, column=1, padx=5, pady=3)
        entries.append(entry)

    def apply_changes():
        new_labels = [e.get() for e in entries]
        plt.legend(new_labels)
        new_title = entry_plot_title.get()
        plt.title(new_title)
        entry_title.delete(0, tk.END)
        entry_title.insert(0, new_title)
        plt.draw()
        editor.destroy()

    tk.Button(editor, text="Apply", command=apply_changes).grid(row=len(current_labels) + 1, columnspan=2, pady=10)


def plot_selected():
    selected = tree.selection()
    if not selected:
        messagebox.showinfo("No selection", "Please select one or more files from the list.")
        return
    plot_files(selected)


def plot_all():
    if not loaded_files:
        messagebox.showinfo("No data", "Load at least one file first.")
        return
    plot_files(loaded_files.keys())


def plot_files(file_paths):
    idx = validate_indices()
    if idx == (None, None):
        return
    try:
        raw_thick = float(entry_raw_thick.get())
        filt_thick = float(entry_filtered_thick.get())
    except ValueError:
        messagebox.showerror("Error", "Line thickness must be numeric.")
        return
    start, end = idx
    plt.figure(figsize=(10, 5))
    raw_lines = []
    filtered_lines = []
    for path in file_paths:
        traces = compute_trace(path, start, end)
        for label, series in traces:
            if "Raw" in label:
                raw_lines.append((label, series))
            else:
                filtered_lines.append((label, series))

    for label, series in raw_lines:
        if isinstance(series, tuple):
            if label == "__zero_cross__":
                plt.plot(series[0], series[1], 'x',
                         color=entry_zero_color.get(),
                         markersize=int(entry_zero_size.get()),
                         markeredgewidth=int(entry_zero_edge.get()))
            else:
                plt.plot(series[0], series[1], 'x', label=label,
                         color=entry_zero_color.get(),
                         markersize=int(entry_zero_size.get()),
                         markeredgewidth=int(entry_zero_edge.get()))
        else:
            plt.plot(series, label=label, linewidth=raw_thick)

    for label, series in filtered_lines:
        if isinstance(series, tuple):
            if label == "__zero_cross__":
                plt.plot(series[0], series[1], 'x',
                         color=entry_zero_color.get(),
                         markersize=int(entry_zero_size.get()),
                         markeredgewidth=int(entry_zero_edge.get()))
            else:
                plt.plot(series[0], series[1], 'x', label=label,
                         color=entry_zero_color.get(),
                         markersize=int(entry_zero_size.get()),
                         markeredgewidth=int(entry_zero_edge.get()))
        else:
            plt.plot(series, label=label, linewidth=filt_thick)

    plt.title(entry_title.get())
    plt.xlabel(entry_xlabel.get())
    plt.ylabel(entry_ylabel.get())
    legend = plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    current_labels = [text.get_text() for text in legend.get_texts()]
    open_legend_editor(current_labels)


def compute_stats_from_magnitude(mag, fs=FS):
    stats = {}
    peaks, _ = find_peaks(mag)
    valleys, _ = find_peaks(-mag)
    peak_values = mag[peaks]
    valley_values = mag[valleys]
    stats['Mean Peak Height'] = np.mean(peak_values) if len(peak_values) > 0 else "N/A"
    stats['Std Peak Height'] = np.std(peak_values) if len(peak_values) > 0 else "N/A"
    stats['Max Peak Height'] = np.max(peak_values) if len(peak_values) > 0 else "N/A"
    stats['Min Peak Height'] = np.min(peak_values) if len(peak_values) > 0 else "N/A"
    stats['Mean Valley Depth'] = np.mean(valley_values) if len(valley_values) > 0 else "N/A"
    stats['Std Valley Depth'] = np.std(valley_values) if len(valley_values) > 0 else "N/A"
    stats['Max Valley Depth'] = np.max(valley_values) if len(valley_values) > 0 else "N/A"
    stats['Min Valley Depth'] = np.min(valley_values) if len(valley_values) > 0 else "N/A"
    stats['RMS Magnitude'] = np.sqrt(np.mean(mag ** 2))
    stats['Peak Count'] = len(peak_values)
    stats['Valley Count'] = len(valley_values)
    stats['Average Peak Distance'] = np.mean(np.diff(peaks)) if len(peaks) > 1 else "N/A"
    stats['Average Valley Distance'] = np.mean(np.diff(valleys)) if len(valleys) > 1 else "N/A"
    stats['Signal Duration (s)'] = len(mag) / fs
    stats['Mean Magnitude'] = np.mean(mag)
    stats['Std Magnitude'] = np.std(mag)
    return stats


def open_stats_window():
    if not loaded_files:
        messagebox.showinfo("No data", "Load at least one file first.")
        return
    start, end = validate_indices()
    if (start, end) == (None, None):
        return
    stats_win = tk.Toplevel()
    stats_win.title("Computed Signal Statistics for All Files")
    stats_win.geometry("500x700")
    text = tk.Text(stats_win, wrap="word")
    text.pack(expand=True, fill="both", padx=10, pady=10)
    for path, display_name in loaded_files.items():
        try:
            with open(path, "r") as f:
                data = json.load(f)
            x = data["linear_acceleration"]["x"][start:end]
            y = data["linear_acceleration"]["y"][start:end]
            z = data["linear_acceleration"]["z"][start:end]
            if end is None:
                end = len(x)
            mag = compute_magnitude(x, y, z)
            stats = compute_stats_from_magnitude(np.array(mag))

            gender = meta.get(path, {}).get("gender", "?")
            height = meta.get(path, {}).get("height", "?")
            speed = meta.get(path, {}).get("speed", "?")
            leg_length = data.get("leg_length", "?")

            text.insert(tk.END, f"File: {display_name}\n")
            text.insert(tk.END, f"  Gender: {gender}\n")
            text.insert(tk.END, f"  Height: {height} m\n")
            text.insert(tk.END, f"  Leg Length: {leg_length} m\n")
            text.insert(tk.END, f"  Walking Speed: {speed}\n")
            text.insert(tk.END, "\n  --- Signal Stats ---\n")
            for k, v in stats.items():
                text.insert(tk.END, f"  {k}: {v}\n")
            text.insert(tk.END, "\n")
        except Exception as e:
            text.insert(tk.END, f"Error processing {display_name}: {e}\n\n")


# ----------------- GUI -----------------

root = tk.Tk()
root.title("Acceleration Visualizer")
root.configure(bg="white")

tk.Label(root, text="Acceleration Visualizer - Data & Filtering", font=("Segoe UI", 14, "bold"), bg="white").pack(pady=(10, 5))

# --------------- Input Range Section ---------------
frame_range = tk.Frame(root, bg="white")
frame_range.pack(pady=6)
tk.Label(frame_range, text="Start Index:", bg="white").grid(row=0, column=0, padx=6)
entry_start = tk.Entry(frame_range, width=8, justify="center")
entry_start.insert(0, "0")
entry_start.grid(row=0, column=1)
tk.Label(frame_range, text="End Index:", bg="white").grid(row=0, column=2, padx=6)
entry_end = tk.Entry(frame_range, width=8, justify="center")
entry_end.insert(0, "800")
entry_end.grid(row=0, column=3)

var_read_all = tk.BooleanVar(value=False)


def toggle_read_all():
    entry_start.config(state="disabled" if var_read_all.get() else "normal")
    entry_end.config(state="disabled" if var_read_all.get() else "normal")


chk_read_all = tk.Checkbutton(frame_range, text="Use Full Range", variable=var_read_all, command=toggle_read_all,
                              bg="white")
chk_read_all.grid(row=0, column=4, padx=10)

# --------------- Filtering Options Section ---------------
frame_filter = tk.Frame(root, bg="white")
frame_filter.pack(pady=6)
var_lowpass = tk.BooleanVar(value=True)
var_highpass = tk.BooleanVar(value=False)
var_show_raw = tk.BooleanVar(value=False)
var_show_components = tk.BooleanVar(value=False)
var_hide_magnitude = tk.BooleanVar(value=False)
# --------------- Zero-crossing Marker Options ---------------
tk.Label(root, text="Zero-Crossing Display Settings", font=("Segoe UI", 10, "bold"), bg="white").pack(pady=(10, 0))

var_zero_crossing = tk.BooleanVar(value=False)
tk.Checkbutton(root, text="Mark Zero-crossings with 'x'", variable=var_zero_crossing, bg="white").pack(pady=2)
frame_zero_x = tk.Frame(root, bg="white")
frame_zero_x.pack(pady=6)

tk.Label(frame_zero_x, text="Zero-cross Marker Color:", bg="white").grid(row=0, column=0, padx=6)
entry_zero_color = tk.Entry(frame_zero_x, width=10, justify="center")
entry_zero_color.insert(0, "red")
entry_zero_color.grid(row=0, column=1)

tk.Label(frame_zero_x, text="Size:", bg="white").grid(row=0, column=2, padx=6)
entry_zero_size = tk.Entry(frame_zero_x, width=6, justify="center")
entry_zero_size.insert(0, "10")
entry_zero_size.grid(row=0, column=3)

tk.Label(frame_zero_x, text="Edge Width:", bg="white").grid(row=0, column=4, padx=6)
entry_zero_edge = tk.Entry(frame_zero_x, width=6, justify="center")
entry_zero_edge.insert(0, "2")
entry_zero_edge.grid(row=0, column=5)

tk.Checkbutton(frame_filter, text="Low‑pass Data:", variable=var_lowpass, bg="white").grid(row=0, column=0, sticky="ew")
lp_frame, entry_lp_cutoff = create_entry_with_hz(frame_filter, "3.0")
lp_frame.grid(row=0, column=1, sticky="ew")
tk.Label(frame_filter, text="Order:", bg="white").grid(row=0, column=2)
entry_lp_order = tk.Entry(frame_filter, width=4, justify="center")
entry_lp_order.insert(0, "4")
entry_lp_order.grid(row=0, column=3)

tk.Checkbutton(frame_filter, text="High‑pass Data:", variable=var_highpass, bg="white").grid(row=1, column=0,
                                                                                             sticky="we", pady=4)

var_kalman = tk.BooleanVar(value=False)
var_moving_avg = tk.BooleanVar(value=False)
kalman_container = tk.Frame(frame_filter, bg="white")
kalman_container.grid(row=3, column=0, columnspan=4, pady=2)
tk.Checkbutton(kalman_container, text="Apply Kalman Filter", variable=var_kalman, bg="white").pack()

tk.Checkbutton(frame_filter, text="Apply Moving Average Filter", variable=var_moving_avg, bg="white").grid(row=4,
                                                                                                           column=0,
                                                                                                           sticky="ew",
                                                                                                           pady=2)
tk.Label(frame_filter, text="MA Window:", bg="white").grid(row=4, column=1, padx=(10, 2), sticky="e")
entry_ma_window = tk.Entry(frame_filter, width=4, justify="center")
entry_ma_window.insert(0, "5")
entry_ma_window.grid(row=4, column=2, sticky="ew")

var_net_mag = tk.BooleanVar(value=False)

netmag_container = tk.Frame(frame_filter, bg="white")
netmag_container.grid(row=5, column=0, columnspan=4, pady=2)
tk.Checkbutton(netmag_container, text="Apply Net Magnitude Filter", variable=var_net_mag, bg="white").pack(side="left")
tk.Label(netmag_container, text="Window:", bg="white").pack(side="left", padx=(10, 2))
entry_net_window = tk.Entry(netmag_container, width=4, justify="center")
entry_net_window.insert(0, "5")
entry_net_window.pack(side="left")

tk.Checkbutton(root, text="Show X/Y/Z components", variable=var_show_components, bg="white").pack(pady=2)
tk.Checkbutton(root, text="Don't show magnitude", variable=var_hide_magnitude, bg="white").pack(pady=2)

var_minmax = tk.BooleanVar(value=False)
tk.Checkbutton(frame_filter, text="Apply Min‑Max Scaling", variable=var_minmax, bg="white").grid(row=6, column=0,
                                                                                                 columnspan=1,
                                                                                                 sticky="w", pady=2)

minmax_frame = tk.Frame(frame_filter, bg="white")
minmax_frame.grid(row=6, column=1, columnspan=3, sticky="w")

tk.Label(minmax_frame, text="Min:", bg="white").pack(side="left", padx=(4, 2))
entry_minmax_min = tk.Entry(minmax_frame, width=5, justify="center")
entry_minmax_min.insert(0, "0")
entry_minmax_min.pack(side="left", padx=(0, 8))

tk.Label(minmax_frame, text="Max:", bg="white").pack(side="left", padx=(4, 2))
entry_minmax_max = tk.Entry(minmax_frame, width=5, justify="center")
entry_minmax_max.insert(0, "1")
entry_minmax_max.pack(side="left")

hp_frame, entry_hp_cutoff = create_entry_with_hz(frame_filter, "0.5")
hp_frame.grid(row=1, column=1, sticky="ew")
tk.Label(frame_filter, text="Order:", bg="white").grid(row=1, column=2)
entry_hp_order = tk.Entry(frame_filter, width=4, justify="center")
entry_hp_order.insert(0, "4")
entry_hp_order.grid(row=1, column=3)

# --------------- Plot Styling Options ---------------
var_split_band = tk.BooleanVar(value=False)
tk.Checkbutton(frame_filter, text="Split LP & HP instead of Band-pass", variable=var_split_band, bg="white").grid(row=2,
                                                                                                                  column=0,
                                                                                                                  columnspan=4,
                                                                                                                  pady=2,
                                                                                                                  sticky="ew")

tk.Label(root, text="Plot Style Settings", font=("Segoe UI", 10, "bold"), bg="white").pack(pady=(10, 0))

tk.Checkbutton(root, text="Show raw data in plot", variable=var_show_raw, bg="white").pack(pady=4)

var_show_name = tk.BooleanVar(value=False)
var_show_stats = tk.BooleanVar(value=True)
frame_labels = tk.Frame(root, bg="white")
frame_labels.pack(pady=4)
tk.Checkbutton(frame_labels, text="Show filename in plot", variable=var_show_name, bg="white").grid(row=0, column=0,
                                                                                                    padx=6)
tk.Checkbutton(frame_labels, text="Show stats in plot", variable=var_show_stats, bg="white").grid(row=0, column=1,
                                                                                                  padx=6)
frame_options = tk.Frame(root, bg="white")
frame_options.pack(pady=10)

frame_line_widths = tk.Frame(frame_options, bg="white")
frame_line_widths.pack(anchor="center", pady=4)
tk.Label(frame_line_widths, text="Raw Line Width:", bg="white").grid(row=0, column=0, sticky="e")
entry_raw_thick = tk.Entry(frame_line_widths, width=6, justify="center")
entry_raw_thick.insert(0, "2")
entry_raw_thick.grid(row=0, column=1, padx=6, pady=2)

tk.Label(frame_line_widths, text="Filtered Line Width:", bg="white").grid(row=0, column=2, sticky="e")
entry_filtered_thick = tk.Entry(frame_line_widths, width=6, justify="center")
entry_filtered_thick.insert(0, "2")
entry_filtered_thick.grid(row=0, column=3, padx=6, pady=2)

frame_row2 = tk.Frame(frame_options, bg="white")
frame_row2.pack(anchor="center", pady=4)
tk.Label(frame_row2, text="Plot Title:", bg="white").grid(row=0, column=0, padx=6, pady=2, sticky="e")
entry_title = tk.Entry(frame_row2, width=40, justify="center")
entry_title.insert(0, "Accelerometer Magnitude Data")
entry_title.grid(row=0, column=1, padx=6, pady=2)

frame_row3 = tk.Frame(frame_options, bg="white")
frame_row3.pack(anchor="center", pady=4)
tk.Label(frame_row3, text="X-Axis label:", bg="white").grid(row=0, column=0, padx=6, pady=2, sticky="e")
entry_xlabel = tk.Entry(frame_row3, width=25, justify="center")
entry_xlabel.insert(0, "Sample index")
entry_xlabel.grid(row=0, column=1, padx=6, pady=2)

frame_row4 = tk.Frame(frame_options, bg="white")
frame_row4.pack(anchor="center", pady=4)
tk.Label(frame_row4, text="Y-Axis label:", bg="white").grid(row=0, column=0, padx=6, pady=2, sticky="e")
entry_ylabel = tk.Entry(frame_row4, width=25, justify="center")
entry_ylabel.insert(0, "Magnitude (m/s²)")
entry_ylabel.grid(row=0, column=1, padx=6, pady=2)

# --------------- Action Buttons Section ---------------
frame_btn = tk.Frame(root, bg="white")
frame_btn.pack(pady=10)
tk.Button(frame_btn, text="Load Files", width=12, command=load_file).grid(row=0, column=0, padx=6)
tk.Button(frame_btn, text="Plot All", width=12, command=plot_all).grid(row=0, column=1, padx=6)
tk.Button(frame_btn, text="Plot Selected", width=14, command=plot_selected).grid(row=0, column=2, padx=6)
tk.Button(frame_btn, text="Compute Stats", width=14, command=open_stats_window).grid(row=0, column=3, padx=6)
tk.Button(frame_btn, text="Clear Selected", width=14, command=clear_selected).grid(row=0, column=4, padx=6)
tk.Button(frame_btn, text="Clear All", width=12, command=clear_all).grid(row=0, column=5, padx=6)

# --------------- Loaded Files Table ---------------
frame_filelist = tk.LabelFrame(root, text="Loaded files", bg="white")
frame_filelist.pack(padx=8, pady=10, fill="both", expand=True)

columns = ("Filename", "Gender", "Height", "Leg Length", "Speed", "Position")
tree = ttk.Treeview(frame_filelist, columns=columns, show="headings", selectmode="extended")
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, anchor="center", width=100)
tree.pack(fill="both", expand=True)

scrollbar = ttk.Scrollbar(frame_filelist, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side="right", fill="y")

root.mainloop()
