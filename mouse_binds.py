import platform
import tkinter as tk

def bind_scroll_events(canvas, scrollable_widget, tree_widget=None):
    """Bind mousewheel scrolling to canvas and (optionally) Treeview widget, cross-platform."""

    def _on_scrollable_mousewheel(event):
        canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def _on_tree_mousewheel(event):
        tree_widget.yview_scroll(-1 * (event.delta // 120), "units")

    # Windows/macOS: bind <MouseWheel>
    if platform.system() in ("Windows", "Darwin"):
        scrollable_widget.bind("<Enter>", lambda e: canvas.master.bind_all("<MouseWheel>", _on_scrollable_mousewheel))
        scrollable_widget.bind("<Leave>", lambda e: canvas.master.unbind_all("<MouseWheel>"))

        if tree_widget:
            tree_widget.bind("<Enter>", lambda e: canvas.master.bind_all("<MouseWheel>", _on_tree_mousewheel))
            tree_widget.bind("<Leave>", lambda e: canvas.master.bind_all("<MouseWheel>", _on_scrollable_mousewheel))

    # Linux/X11: bind Button-4 and Button-5
    elif platform.system() == "Linux":
        scrollable_widget.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        scrollable_widget.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        if tree_widget:
            tree_widget.bind("<Button-4>", lambda e: tree_widget.yview_scroll(-1, "units"))
            tree_widget.bind("<Button-5>", lambda e: tree_widget.yview_scroll(1, "units"))