import tkinter as tk
from tkinter import ttk
from jse.core.instruments import Instrument

class TickerInput(ttk.Frame):
    """Composite widget: entry + validate + callback."""
    def __init__(self, master, on_add, placeholder="Type symbol, e.g. AAPL"):
        super().__init__(master)
        self.on_add = on_add
        self.var = tk.StringVar()
        ttk.Entry(self, textvariable=self.var, width=20).pack(side=tk.LEFT, padx=2)
        ttk.Button(self, text="➕ Add", command=self._add).pack(side=tk.LEFT, padx=2)
        self.var.set(placeholder)
        self.var.trace_add("write", lambda *a: self._colour_placeholder())

    def _colour_placeholder(self):
        colour = "grey" if self.var.get() == self.var.get() else "black"
        self.nametowidget(self.winfo_children()[0]).config(foreground=colour)

    def _add(self):
        sym = self.var.get().strip().upper()
        if not sym:
            return
        try:
            Instrument.fetch(sym)   # raises if Yahoo unknown
            self.on_add(sym)
            self.var.set("")
        except Exception as e:
            tk.messagebox.showerror("Invalid ticker", str(e))
