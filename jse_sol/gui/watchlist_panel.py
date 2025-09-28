import tkinter as tk
from tkinter import ttk
from jse.core.portfolio import WatchList

class WatchListPanel(ttk.LabelFrame):
    def __init__(self, master, change_callback):
        super().__init__(master, text="Watch-list", padding=5)
        self.change_cb = change_callback
        self.listbox = tk.Listbox(self, height=15, selectmode=tk.SINGLE, activestyle="none")
        self.listbox.pack(fill=tk.BOTH, expand=True)
        ttk.Button(self, text="➖ Remove", command=self._remove).pack(fill=tk.X, pady=2)
        self.listbox.bind("<Delete>", lambda e: self._remove())
        # enable drag-drop reorder
        self.listbox.bind("<Button-1>", self._start_drag)
        self.listbox.bind("<B1-Motion>", self._drag)
        self._populate()

    # ---------- internals ----------
    def _populate(self):
        self.listbox.delete(0, tk.END)
        for ins in WatchList().items:
            self.listbox.insert(tk.END, f"{ins.symbol} – {ins.name}")

    def _remove(self):
        idx = self.listbox.curselection()
        if not idx:
            return
        sym = WatchList().items[idx[0]].symbol
        WatchList().remove(sym)
        self._populate()
        self.change_cb()

    # ---------- drag-drop ----------
    def _start_drag(self, event):
        self.drag_idx = self.listbox.nearest(event.y)

    def _drag(self, event):
        new_idx = self.listbox.nearest(event.y)
        if new_idx != self.drag_idx:
            wl = WatchList()
            wl.items.insert(new_idx, wl.items.pop(self.drag_idx))
            wl.save()
            self.drag_idx = new_idx
            self._populate()
            self.change_cb()

    # ---------- external add ----------
    def add_symbol(self, symbol: str):
        if WatchList().add(symbol):
            self._populate()
            self.change_cb()
