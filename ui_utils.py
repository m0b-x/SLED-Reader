
import tkinter as tk

def create_entry_with_hz(parent, default_value=""):
    frame = tk.Frame(parent, bg="white")
    entry = tk.Entry(frame, width=6, justify="center")
    entry.insert(0, default_value)
    entry.pack(side="left")
    tk.Label(frame, text="Hz,", bg="white").pack(side="left", padx=(2, 0))
    return frame, entry



def create_expandable_section(parent, title, big=False):
    container = tk.Frame(parent, bg="white")
    header    = tk.Frame(container, bg="white")
    header.pack(fill="x")

    mid = tk.Frame(header, bg="white")
    mid.pack(expand=True)

    is_expanded = tk.BooleanVar(value=True)
    font_size   = 14 if big else 10

    arrow = tk.Label(mid, text="▼",
                     font=("Segoe UI", font_size, "bold"),
                     bg="white")
    arrow.pack(side="left")

    lbl = tk.Label(mid, text=title,
                   font=("Segoe UI", font_size, "bold"),
                   bg="white")
    lbl.pack(side="left", padx=(2, 0))

    def toggle(_=None):
        if is_expanded.get():
            content.pack_forget()
            arrow.config(text="🞂")
            is_expanded.set(False)
        else:
            content.pack(fill="x", padx=8, pady=4)
            arrow.config(text="▼")
            is_expanded.set(True)

    # make entire header clickable
    for w in (header, mid, arrow, lbl):
        w.bind("<Button-1>", toggle)

    content = tk.Frame(container, bg="white")
    content.pack(fill="x", padx=8, pady=4)
    return container, content
