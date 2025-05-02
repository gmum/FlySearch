from PIL import Image
from typing import Dict

from scenarios.city_scenario_mapper import CityScenarioMapper

classes_to_images = {
    CityScenarioMapper.ObjectType.FIRE: Image.open("web/client/public/targets/city_fire_1.png.jpg"),

    CityScenarioMapper.ObjectType.CROWD: Image.open("web/client/public/targets/crowd.png.jpg"),

    CityScenarioMapper.ObjectType.POLICE_CAR: Image.open("web/client/public/targets/police_car.png.jpg"),
    CityScenarioMapper.ObjectType.WHITE_SPORT_CAR: Image.open("web/client/public/targets/white_sport_car.png.jpg"),
    CityScenarioMapper.ObjectType.RED_SPORT_CAR: Image.open("web/client/public/targets/red_sport_car.png.jpg"),
    CityScenarioMapper.ObjectType.BLUE_SPORT_CAR: Image.open("web/client/public/targets/blue_sport_car.png.jpg"),
    CityScenarioMapper.ObjectType.BEIGE_SPORT_CAR: Image.open("web/client/public/targets/beige_sport_car.png.jpg"),

    CityScenarioMapper.ObjectType.BLACK_TRUCK: Image.open("web/client/public/targets/black_truck.png.jpg"),
    CityScenarioMapper.ObjectType.WHITE_TRUCK: Image.open("web/client/public/targets/white_truck.png.jpg"),

    CityScenarioMapper.ObjectType.RED_PICKUP_TRUCK: Image.open("web/client/public/targets/red_pickup_truck.png.jpg"),
    CityScenarioMapper.ObjectType.GREEN_PICKUP_TRUCK: Image.open(
        "web/client/public/targets/green_pickup_truck.png.jpg"),
    CityScenarioMapper.ObjectType.BLACK_PICKUP_TRUCK: Image.open(
        "web/client/public/targets/black_pickup_truck.png.jpg"),
    CityScenarioMapper.ObjectType.WHITE_PICKUP_TRUCK: Image.open(
        "web/client/public/targets/white_pickup_truck.png.jpg"),

    CityScenarioMapper.ObjectType.ROAD_CONSTRUCTION_SITE: Image.open(
        "web/client/public/targets/construction_works_1.png.jpg"),

    CityScenarioMapper.ObjectType.LARGE_TRASH_PILE: Image.open("web/client/public/targets/large_trash_pile.png.jpg"),
}
