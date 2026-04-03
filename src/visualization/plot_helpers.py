from matplotlib import colormaps as cm
from matplotlib.colors import ListedColormap
from numpy import linspace

def get_color_map():
    viridisBig = cm.get_cmap('jet')
    newcmp = ListedColormap(viridisBig(linspace(0, 1, 15)))
    return newcmp