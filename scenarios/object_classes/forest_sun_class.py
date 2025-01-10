class ForestSunClass:
    def __init__(self, sun_id, client):
        self.sun_id = sun_id
        self.client = client

    def set_sun_rotation(self, sun_y, sun_z):
        self.client.request(f"vset /object/{self.sun_id}/rotation 0 {sun_y} {sun_z}")
