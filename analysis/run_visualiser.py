import numpy as np
import math
import seaborn as sns

from string import ascii_uppercase, ascii_lowercase
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from analysis import Run


class RunVisualiser:
    def __init__(self, run: Run):
        self.run = run

    def plot_tikz(self) -> str:
        coords = self.run.get_coords()

        x = [coord[0] for coord in coords]
        y = [coord[1] for coord in coords]
        z = [coord[2] for coord in coords]

        max_x = max(max(x), 0)
        max_y = max(max(y), 0)
        max_z = max(max(z), 0)

        min_x = min(min(x), 0)
        min_y = min(min(y), 0)
        min_z = min(min(z), 0)

        end_location = coords[-1]
        end_x = end_location[0]
        end_y = end_location[1]
        end_z = end_location[2]

        max_x = max(max_x, end_x + end_z)
        max_y = max(max_y, end_y + end_z)

        min_x = min(min_x, end_x - end_z)
        min_y = min(min_y, end_y - end_z)

        total_max = max(max_x, max_y, max_z)
        total_min = min(min_x, min_y, min_z)

        margin = 1

        total_max += margin
        total_min -= margin

        scaling_ratio = 2 / (total_max - total_min)

        def to_tikz_coords(x1, y1, z1):
            x1 = -1 + scaling_ratio * (x1 - total_min)
            y1 = -1 + scaling_ratio * (y1 - total_min)
            z1 = scaling_ratio * z1

            return x1, y1, z1

        for i, ltr in zip(range(len(x) - 1), ascii_uppercase):
            x_curr = x[i]
            y_curr = y[i]
            z_curr = z[i]

            display_coords = (x_curr, y_curr, z_curr)

            x_next = x[i + 1]
            y_next = y[i + 1]
            z_next = z[i + 1]

            display = True

            if ((x_curr - x_next) ** 2 + (y_curr - y_next) ** 2 + (z_curr - z_next) ** 2) ** 0.5 < 10:
                display = False

            if x_curr == x_next and y_curr == y_next and z_curr == z_next:
                continue

            x_curr, y_curr, z_curr = to_tikz_coords(x_curr, y_curr, z_curr)
            x_next, y_next, z_next = to_tikz_coords(x_next, y_next, z_next)

            start_coords = (x_curr, y_curr, z_curr)
            next_coords = (x_next, y_next, z_next)

            if display:
                print(
                    f"\draw[flight] {start_coords} node [right] {'{\\tiny \(' + ltr + '.' + str(display_coords) + '\)}'} -- {next_coords} ;")
            else:
                print(
                    f"\draw[flight] {start_coords} node [right] {'{\\tiny \(' + ltr + '.\)}'} -- {next_coords} ;")

        viewcone = [
            (end_x + end_z, end_y + end_z, 0),
            (end_x - end_z, end_y + end_z, 0),
            (end_x - end_z, end_y - end_z, 0),
            (end_x + end_z, end_y - end_z, 0),
        ]

        end_x, end_y, end_z = to_tikz_coords(end_x, end_y, end_z)

        print(
            f"\draw[dist] {end_x, end_y, end_z} node [right] {'{\\tiny \(' + ascii_uppercase[len(x) - 1] + '.' + str(end_location) + '\)}'} -- " + str(
                to_tikz_coords(0, 0, 0)) + ";")

        viewcone = [to_tikz_coords(*coords) for coords in viewcone]

        for coords in viewcone:
            print(f"\draw[viewcone] {coords} -- ({end_x, end_y, end_z});")

        # \draw[fill = blue, opacity = 0.2] (1, -1, 0) - - (1, -1, 2) - - (1, 1, 2) - - (1, 1, 0) - - cycle;

        print("\draw[fill = green, opacity = 0.4]", viewcone[0], "--", viewcone[1], "--", viewcone[2], "--",
              viewcone[3],
              "-- cycle;")

        # \node[draw=none,shape=circle,fill, inner sep=2pt, color=red] (d1) at (0,0,0){};  % circle

        print(r"\node[draw=none,shape=circle,fill, inner sep=2pt, color=red] (d1) at" + str(
            to_tikz_coords(0, 0, 0)) + "{};")

        """
        \draw[axis](0, 0, 0) - - (0, 0, 2)
        node[right]
        {\scriptisze \(Z\)};
        \draw[axis](-1, 0, 0) - - (1, 0, 0)
        node[right]
        {\scriptisze \(X\)};
        \draw[axis](0, -1, 0) - - (0, 1, 0)
        node[right]
        {\scriptisze \(Y\)};
        """

    def plot_situation_awareness_chart(self, ax):
        coords = self.run.get_coords()

        img_px_h = 500
        img_px_w = 500

        img_px_area = img_px_h * img_px_w

        # This code assumes camera FOV = 90 degrees
        areas = []

        for (x, y, z) in coords:
            min_x = x - z
            max_x = x + z

            min_y = y - z
            max_y = y + z

            # px_per_msq = math.log((img_px_area / (z * z)), 2)
            rpx_per_msq = (img_px_area / (z * z)) ** 0.5

            areas.append((rpx_per_msq, min_x, max_x, min_y, max_y))

        areas = sorted(areas, key=lambda x: x[0])

        margin = 10

        min_x_global = min([area[1] for area in areas])
        max_x_global = max([area[2] for area in areas])

        min_y_global = min([area[3] for area in areas])
        max_y_global = max([area[4] for area in areas])

        min_x_global -= margin
        max_x_global += margin

        min_y_global -= margin
        max_y_global += margin

        x_len = max_x_global - min_x_global
        y_len = max_y_global - min_y_global

        x_len = int(math.ceil(x_len))
        y_len = int(math.ceil(y_len))

        # Create a numpy array to hold px_per_msq values

        px_per_msq_array = np.zeros((x_len, y_len))

        for area in areas:
            rpx_per_msq, min_x, max_x, min_y, max_y = area

            min_x = int(min_x - min_x_global)
            max_x = int(max_x - min_x_global)
            min_y = int(min_y - min_y_global)
            max_y = int(max_y - min_y_global)

            px_per_msq_array[min_x:max_x, min_y:max_y] = rpx_per_msq

        xticklabels = list(range(int(min_x_global), int(max_x_global) + 1))
        yticklabels = list(range(int(min_y_global), int(max_y_global) + 1))

        xticklabels = [str(x) if x % 10 == 0 else "" for x in xticklabels]
        yticklabels = [str(y) if y % 10 == 0 else "" for y in yticklabels]

        sns.heatmap(px_per_msq_array, cbar_kws={'label': 'Root pixels per square meter'}, robust=False, ax=ax,
                    xticklabels=xticklabels, yticklabels=yticklabels)

        ax.set_title("Situation Awareness Chart")

    def plot(self, ax: Axes3D):
        coords = self.run.get_coords()
        x = [coord[0] for coord in coords]
        y = [coord[1] for coord in coords]
        z = [coord[2] for coord in coords]

        ax.scatter(x, y, z, label="Drone path")
        ax.scatter(0, 0, 0, label="Object", color='r')

        # Set vertical axis lim min to 0
        ax.set_zlim([0, max(z)])

        # Draw line from (x1, y1, z1) to (x2, y2, z2)

        for i, ltr in zip(range(len(x) - 1), ascii_uppercase):
            x_curr = x[i]
            y_curr = y[i]
            z_curr = z[i]

            x_next = x[i + 1]
            y_next = y[i + 1]
            z_next = z[i + 1]

            if x_curr == x_next and y_curr == y_next and z_curr == z_next:
                continue

            x_middle = (x_curr + x_next) / 2
            y_middle = (y_curr + y_next) / 2
            z_middle = (z_curr + z_next) / 2

            dx = x_next - x_curr
            dy = y_next - y_curr
            dz = z_next - z_curr

            dx /= 3
            dy /= 3
            dz /= 3

            x_quiv_start = x_curr * (2 / 3) + x_next * (1 / 3)
            y_quiv_start = y_curr * (2 / 3) + y_next * (1 / 3)
            z_quiv_start = z_curr * (2 / 3) + z_next * (1 / 3)

            ax.plot([x_curr, x_next], [y_curr, y_next], [z_curr, z_next], color="black")

            # ax.quiver(x_quiv_start, y_quiv_start, z_quiv_start, dx, dy, dz, color='black',
            #          normalize=False)

            ax.quiver(x_next, y_next, z_next, dx * 3, dy * 3, dz * 3, color='black', normalize=True, pivot="tip",
                      arrow_length_ratio=4)

            # Annotate current point
            ax.text(x_curr + 0.6, y_curr, z_curr, f"{ltr}. ({x_curr}, {y_curr}, {z_curr})")
            ax.scatter(x_curr, y_curr, z_curr, color='b', s=100)

        end_location = coords[-1]
        end_x = end_location[0]
        end_y = end_location[1]
        end_z = end_location[2]

        # Annotate end point
        ax.text(end_x + 0.6, end_y, end_z, f"{ascii_uppercase[len(x) - 1]}. ({end_x}, {end_y}, {end_z})")
        ax.scatter(end_x, end_y, end_z, color='b', s=100)

        ax.text(0.3, 0, 0, "(0, 0, 0)")
        ax.scatter(0, 0, 0, color='r', s=100)

        # Dashed line from end location to origin
        ax.plot([end_x, 0], [end_y, 0], [end_z, 0], color='g', linestyle='dashed', label="Distance to object")

        # View cone from end location

        # fov = 90 degrees
        x_max = end_x + end_z
        x_min = end_x - end_z

        y_min = end_y - end_z
        y_max = end_y + end_z

        ax.plot([end_x, x_min], [end_y, y_min], [end_z, 0], color='r', linestyle='dashed')
        ax.plot([end_x, x_max], [end_y, y_min], [end_z, 0], color='r', linestyle='dashed')
        ax.plot([end_x, x_max], [end_y, y_max], [end_z, 0], color='r', linestyle='dashed')
        ax.plot([end_x, x_min], [end_y, y_max], [end_z, 0], color='r', linestyle='dashed')

        ax.plot([x_min, x_min], [y_min, y_max], [0, 0], color='r', linestyle='dashed')
        ax.plot([x_max, x_max], [y_min, y_max], [0, 0], color='r', linestyle='dashed')
        ax.plot([x_min, x_max], [y_min, y_min], [0, 0], color='r', linestyle='dashed')
        ax.plot([x_min, x_max], [y_max, y_max], [0, 0], color='r', linestyle='dashed', label="View cone")

        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()
        z_ticks = ax.get_zticks()

        x_min = min(x_ticks)
        x_max = max(x_ticks)

        y_min = min(y_ticks)
        y_max = max(y_ticks)

        z_min = min(z_ticks)
        z_max = max(z_ticks)

        x_min, x_max = x_min - 2, x_max + 2
        y_min, y_max = y_min - 2, y_max + 2
        z_min, z_max = z_min - 2, z_max + 2

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        ax.set_aspect("equal")

        ax.legend()

        ax.set_zlabel('Z')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')


def main():
    import os
    from pathlib import Path

    base_path = Path("../all_logs/GPT4o-CityNew")
    runs = sorted(os.listdir(base_path), key=lambda x: int(x.split("_")[0]))

    run = Run(base_path / runs[1])

    visualiser = RunVisualiser(run)

    fig, ax = plt.subplots()
    visualiser.plot_situation_awareness_chart(ax)

    plt.show()

    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax)

    visualiser.plot(ax)
    plt.show()

    # bad idea
    # visualiser.plot_tikz()


if __name__ == '__main__':
    main()
