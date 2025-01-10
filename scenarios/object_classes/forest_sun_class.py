class ForestSunClass:
    def __init__(self, sun_id, client):
        self.sun_id = sun_id
        self.client = client

    def set_sun_rotation(self, sun_y, sun_z):
        # OpenCV has a different coordinate system than that of Unreal's editor.
        # This means a rather awkward placement of y and z coordinates.
        self.client.request(f"vset /object/{self.sun_id}/rotation {sun_y} {sun_z} 0")
