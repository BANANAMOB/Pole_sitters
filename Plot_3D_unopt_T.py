from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pickle

mu = 0.012150585609624

if __name__ == "__main__":

    # PARAMETERS
    T0, Tf = 0.05, 6
    z0, zf = 0.1, 0.5
    acc_z0, acc_zf = 0.2, 2.5

    # UNPACK
    filepath_all = f"DATA/Unoptimized_all_T={T0}-{Tf}_z={z0}-{zf}_acc_z={acc_z0}-{acc_zf}.pkl"
    with open(filepath_all, 'rb') as f:
        orbits_all = pickle.load(f)

    print(f"Loaded: {len(orbits_all)} orbits")

    # FILTER
    filtered_orbits = []
    for ref_pole, state0, T, acc_z in orbits_all:
        is_hover = all(y < 1e-3 for y in ref_pole[1])
        x_lim = all(0 < x < 5 for x in ref_pole[0])
        z_lim = all(z > 0 for z in ref_pole[2])
        if T > 0.05 and x_lim and not is_hover and z_lim:
            filtered_orbits.append([ref_pole, state0, T, acc_z])

    print(f"Filtered: {len(filtered_orbits)} / {len(orbits_all)}")

    # NORM
    min_T = min(o[2] for o in filtered_orbits)
    max_T = max(o[2] for o in filtered_orbits)
    norm = mcolors.Normalize(vmin=min_T, vmax=max_T)
    cmap = cm.plasma
    print(f"Period range: {min_T:.3f} to {max_T:.3f}")

    # STYLE
    style_path = "/home/lha43/Desktop/Lucas/Generated_figures/paper.mplstyle"
    plt.style.use(style_path)

    # FIGURE
    fig = plt.figure(figsize=(12, 10))
    fig.subplots_adjust(bottom=0.12, hspace=0.35, wspace=0.35)

    # 2D VIEWS 
    views_2d = [
        ('Top (XY)',   0, 1, 1),   # title, x-ind
        ('Front (XZ)', 0, 2, 2),
        ('Side (YZ)',  1, 2, 3),
    ]
    labels = {0: '$x$ [DU]', 1: '$y$ [DU]', 2: '$z$ [DU]'}

    for title, xi, yi, pos in views_2d:
        ax = fig.add_subplot(2, 2, pos)
        for ref_pole, state0, T, acc_z in filtered_orbits:
            color = cmap(norm(T))
            ax.plot(ref_pole[xi], ref_pole[yi],
                    alpha=0.8, marker='o', markersize=2, color=color)
        ax.scatter(1 - mu if xi == 0 else 0,
                   0 if yi != 1 else 0,
                   color='gray', s=20, zorder=5, label='Moon')
        ax.set_xlabel(labels[xi])
        ax.set_ylabel(labels[yi])
        #ax.set_title(title, fontsize=10)
        ax.set_axisbelow(True)
        ax.grid(True)
        #ax.legend(fontsize=7)

    # ISOMETRIC 
    ax3d = fig.add_subplot(2, 2, 4, projection='3d')
    ax3d.view_init(elev=25, azim=-45)
    ax3d.scatter(1 - mu, 0, 0, color='gray', s=20, label='Moon')

    for ref_pole, state0, T, acc_z in filtered_orbits:
        color = cmap(norm(T))
        ax3d.plot(ref_pole[0], ref_pole[1], ref_pole[2],
                  alpha=0.8, marker='o', markersize=2, color=color)

    ax3d.set_xlabel('$x$ [DU]', labelpad=10)
    ax3d.set_ylabel('$y$ [DU]', labelpad=10)
    ax3d.set_zlabel('$z$ [DU]', labelpad=3)
    #ax3d.set_title('Isometric', fontsize=10)
    ax3d.legend()

    # COLORBAR  i hate this
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    plt.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm),
                 cax=cbar_ax, orientation='horizontal', label='$T$ [TU]')

    plt.show()
