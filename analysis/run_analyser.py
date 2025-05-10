class RunAnalyser:
    def __init__(self, run):
        self.run = run

    # Assumes that object is at (0, 0, 0)
    def get_euclidean_distance(self) -> float:
        end_coords = self.run.end_position

        return (end_coords[0] ** 2 + end_coords[1] ** 2 + end_coords[2] ** 2) ** 0.5

    def drone_within_altitude_threshold(self, threshold) -> bool:
        bounding_box = self.run.object_bbox
        z_max = bounding_box[5] // 100  # Semi-centimeters to semi-meters

        if "fire" in self.run.object_type.lower():
            z_max = 3

        drone_z = self.run.end_position[2]

        return drone_z - z_max <= threshold

    # Assumes object is at (0, 0, 0)
    def object_visible(self) -> bool:
        drone_x, drone_y, drone_z = self.run.end_position

        # FOV = 90 degress (45 degrees to each side)
        x_diff_visible = drone_z
        y_diff_visible = drone_z

        x_min = drone_x - x_diff_visible
        x_max = drone_x + x_diff_visible

        y_min = drone_y - y_diff_visible
        y_max = drone_y + y_diff_visible

        return x_min <= 0 <= x_max and y_min <= 0 <= y_max

    def success_criterion_satisfied(self, threshold: float = 10) -> bool:
        return self.object_visible() and self.drone_within_altitude_threshold(threshold)
