import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from analysis import Run


class RunVisualiser:
    def __init__(self, run: Run):
        self.run = run

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

        for i in range(len(x) - 1):
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

            dx /= 4
            dy /= 4
            dz /= 4

            ax.plot([x_curr, x_next], [y_curr, y_next], [z_curr, z_next])
            ax.quiver(x_middle, y_middle, z_middle, dx, dy, dz, color='b',
                      normalize=False)

            # Annotate current point
            ax.text(x_curr + 0.3, y_curr, z_curr, f"({x_curr}, {y_curr}, {z_curr})")

        end_location = coords[-1]
        end_x = end_location[0]
        end_y = end_location[1]
        end_z = end_location[2]

        # Annotate end point
        ax.text(end_x + 0.3, end_y, end_z, f"({end_x}, {end_y}, {end_z})")
        ax.text(0.3, 0, 0, "(0, 0, 0)")

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

        ax.legend()


def main():
    import os
    from pathlib import Path

    base_path = Path("../all_logs/MC-0S-F")
    runs = sorted(os.listdir(base_path), key=lambda x: int(x.split("_")[0]))

    run = Run(base_path / runs[3])

    visualiser = RunVisualiser(run)

    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax)

    visualiser.plot(ax)
    plt.show()


if __name__ == '__main__':
    main()
