import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def draw_vector(v, ax, s):
    if len(v) == 2: # wykres 2d
        ax.arrow(
            0, 0, *v,
            head_width=0.03, head_length=0.1, length_includes_head=True)
    else: # wykres 3d
        a = Arrow3D(
            [0, v[0]], [0, v[1]], [0, v[2]],
            mutation_scale=20, lw=3, arrowstyle="-|>")
        ax.add_artist(a)
    ax.text(*v, s=s)
