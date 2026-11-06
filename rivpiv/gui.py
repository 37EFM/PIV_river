"""Small matplotlib tools for marking things on a frame.

``ShapePicker`` lets the user click on an image to draw

* ``'segment'``   -- two clicks per segment, then a length in metres is
                     typed in the text box and confirmed with Enter;
* ``'rectangle'`` -- two clicks per rectangle (opposite corners);
* ``'polygon'``   -- one click per vertex, closed with the button or
                     with the *c* key.

It works with the interactive backends in a notebook (``%matplotlib
widget``, which needs ``ipympl``) and in a window (``%matplotlib qt`` or
a plain script).  Clicks made while the toolbar's zoom or pan tool is
active are ignored, so zooming in to place a point precisely is fine.
Press *u* to undo the last point.

Typical use in a notebook::

    picker = ShapePicker(frames[0], 'segment')
    # ... click, type lengths ...
    picker.result        # -> [[x1, y1, x2, y2, L], ...]
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox


_HELP = {
    "segment": "Click the two ends of a segment on the water surface, then type its length in metres and press Enter.",
    "rectangle": "Click two opposite corners of each rectangle on a static part of the scene (banks).",
    "polygon": "Click the vertices of a polygon around the water, then press 'Close polygon' (or the c key).",
}


class ShapePicker:
    def __init__(self, image, kind="segment", initial=None, figsize=(7, 10), cmap="gray", title=None):
        if kind not in _HELP:
            raise ValueError("kind must be 'segment', 'rectangle' or 'polygon'")
        self.kind = kind
        self.shapes = []           # completed shapes (lists of numbers)
        self.pending = []          # points of the shape being drawn
        self._artists = []         # artists of completed shapes
        self._pending_artists = []

        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_axes([0.05, 0.12, 0.9, 0.84])
        self.ax.imshow(image, cmap=cmap if np.ndim(image) == 2 else None)
        self.ax.set_title(title or _HELP[kind], fontsize=9, wrap=True)
        self.ax.set_xlim(-0.5, np.shape(image)[1] - 0.5)
        self.ax.set_ylim(np.shape(image)[0] - 0.5, -0.5)

        self.msg = self.fig.text(0.05, 0.085, "", fontsize=9)
        if kind == "segment":
            self.box = TextBox(self.fig.add_axes([0.25, 0.02, 0.25, 0.05]), "length (m) ", initial="")
            self.box.on_submit(self._submit_length)
        else:
            self.box = None
        if kind == "polygon":
            self.btn_close = Button(self.fig.add_axes([0.52, 0.02, 0.2, 0.05]), "Close polygon")
            self.btn_close.on_clicked(lambda ev: self._close_polygon())
        self.btn_undo = Button(self.fig.add_axes([0.74, 0.02, 0.1, 0.05]), "Undo")
        self.btn_undo.on_clicked(lambda ev: self.undo())
        self.btn_clear = Button(self.fig.add_axes([0.85, 0.02, 0.1, 0.05]), "Clear")
        self.btn_clear.on_clicked(lambda ev: self.clear())

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        if initial:
            for s in initial:
                flat = [v for item in s for v in (item if np.iterable(item) else [item])]
                flat = [None if v is None else float(v) for v in flat]
                if kind == "segment" and len(flat) == 4:
                    flat.append(None)
                self._add_shape(flat)
        self._say("ready")

    # -- results --------------------------------------------------------
    @property
    def result(self):
        """Completed shapes.  Segments without a length are left out."""
        if self.kind == "segment":
            return [s for s in self.shapes if len(s) == 5 and s[4] is not None]
        return [list(s) for s in self.shapes]

    # -- interaction ----------------------------------------------------
    def _toolbar_busy(self):
        tb = getattr(self.fig.canvas, "toolbar", None)
        return bool(tb is not None and getattr(tb, "mode", ""))

    def _on_click(self, ev):
        if ev.inaxes is not self.ax or ev.button != 1 or self._toolbar_busy():
            return
        self.pending.append((ev.xdata, ev.ydata))
        self._draw_pending()
        if self.kind in ("segment", "rectangle") and len(self.pending) == 2:
            (x1, y1), (x2, y2) = self.pending
            self.pending = []
            self._clear_pending_artists()
            if self.kind == "segment":
                self._add_shape([x1, y1, x2, y2, None])
                self._say("now type the length of this segment in metres and press Enter")
            else:
                self._add_shape([x1, y1, x2, y2])
        self.fig.canvas.draw_idle()

    def _on_key(self, ev):
        if ev.key == "u":
            self.undo()
        elif ev.key == "c" and self.kind == "polygon":
            self._close_polygon()

    def _submit_length(self, text):
        try:
            L = float(text.replace(",", "."))
        except ValueError:
            self._say(f"could not read a number from '{text}'")
            return
        for k in range(len(self.shapes) - 1, -1, -1):
            if self.shapes[k][4] is None:
                break
        else:
            k = len(self.shapes) - 1 if self.shapes else None
        if k is None:
            self._say("draw a segment first")
            return
        self.shapes[k][4] = L
        self._artists[k][-1].set_text(f"{L:g} m")
        self.box.set_val("")
        self._say(f"segment {k + 1}: {L:g} m.  {len(self.result)} segment(s) with a length.")
        self.fig.canvas.draw_idle()

    def _close_polygon(self):
        if len(self.pending) < 3:
            self._say("a polygon needs at least three vertices")
            return
        pts = [v for p in self.pending for v in p]
        self.pending = []
        self._clear_pending_artists()
        self._add_shape(pts)
        self.fig.canvas.draw_idle()

    def undo(self):
        if self.pending:
            self.pending.pop()
            self._draw_pending()
        elif self.shapes:
            self.shapes.pop()
            for a in self._artists.pop():
                a.remove()
        self._say("undone")
        self.fig.canvas.draw_idle()

    def clear(self):
        self.pending = []
        self._clear_pending_artists()
        while self.shapes:
            self.shapes.pop()
            for a in self._artists.pop():
                a.remove()
        self._say("cleared")
        self.fig.canvas.draw_idle()

    # -- drawing --------------------------------------------------------
    def _say(self, text):
        self.msg.set_text(text)
        self.fig.canvas.draw_idle()

    def _clear_pending_artists(self):
        for a in self._pending_artists:
            a.remove()
        self._pending_artists = []

    def _draw_pending(self):
        self._clear_pending_artists()
        if self.pending:
            p = np.array(self.pending)
            self._pending_artists += self.ax.plot(p[:, 0], p[:, 1], "o-", color="yellow", ms=5, lw=1)

    def _add_shape(self, s):
        k = len(self.shapes)
        self.shapes.append(s)
        colour = plt.cm.tab10(k % 10)
        arts = []
        if self.kind == "segment":
            x1, y1, x2, y2 = s[:4]
            arts += self.ax.plot([x1, x2], [y1, y2], "o-", color=colour, ms=5, lw=1.5)
            label = f"{s[4]:g} m" if len(s) > 4 and s[4] is not None else "?"
            arts.append(self.ax.text((x1 + x2) / 2, (y1 + y2) / 2, label, color=colour,
                                     fontsize=10, fontweight="bold",
                                     bbox=dict(facecolor="black", alpha=0.5, pad=1, lw=0)))
        elif self.kind == "rectangle":
            x1, y1, x2, y2 = s
            arts += self.ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "-", color=colour, lw=1.5)
            arts.append(self.ax.text(min(x1, x2), min(y1, y2), f" {k + 1}", color=colour, va="bottom"))
        else:
            p = np.array(s).reshape(-1, 2)
            p = np.vstack([p, p[:1]])
            arts += self.ax.plot(p[:, 0], p[:, 1], "o-", color=colour, ms=4, lw=1.5)
        self._artists.append(arts)


def draw_shapes(ax, shapes, kind, color="yellow", **kw):
    """Draw saved shapes (segments, rectangles or polygons) on an axes."""
    for k, s in enumerate(shapes):
        if kind == "segment":
            ax.plot([s[0], s[2]], [s[1], s[3]], "o-", color=color, ms=4, lw=1.5, **kw)
            if len(s) > 4 and s[4] is not None:
                ax.text((s[0] + s[2]) / 2, (s[1] + s[3]) / 2, f"{s[4]:g} m", color=color,
                        fontsize=9, fontweight="bold", bbox=dict(facecolor="black", alpha=0.5, pad=1, lw=0))
        elif kind == "rectangle":
            x1, y1, x2, y2 = s
            ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "-", color=color, lw=1.5, **kw)
        else:
            p = np.array(s, float).reshape(-1, 2)
            p = np.vstack([p, p[:1]])
            ax.plot(p[:, 0], p[:, 1], "-", color=color, lw=1.5, **kw)
