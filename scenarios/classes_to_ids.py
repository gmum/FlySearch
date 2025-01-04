from scenarios.city_scenario_mapper import CityScenarioMapper
from scenarios.forest_scenario_mapper import ForestScenarioMapper

from scenarios.object_classes import BaseObjectClass, PCGClass, EnvPCGClass, ForestEnvPCGClass
from scenarios.object_classes.forest_sun_class import ForestSunClass


def get_classes_to_object_classes(client):
    # Forest
    forest_fire = BaseObjectClass(["NiagaraActor_UAID_08BFB8191750822F02_1834612234"], client=client)
    forest_trash = BaseObjectClass(["Actor_UAID_08BFB8191750053702_1677020459"], client=client)
    camping = PCGClass("BP_RandomSpawner_C_UAID_08BFB8191750113C02_1871747848",
                       ["StaticMeshActor_UAID_08BFB81917501C3F02_1703388951"], client=client)
    building = BaseObjectClass(["StaticMeshActor_UAID_08BFB8191750802F02_1934372840"], client=client)
    person = BaseObjectClass(["SkeletalMeshActor_UAID_08BFB8191750A22F02_1368103644"], client=client)
    ufo = BaseObjectClass(["Actor_UAID_08BFB8191750053702_1321019457"], client=client)
    plane = BaseObjectClass(["Actor_UAID_08BFB8191750043702_2140437280"], client=client)
    helicopter = BaseObjectClass(["StaticMeshActor_UAID_08BFB8191750683002_1545951497"], client=client)
    sun = ForestSunClass("DirectionalLight_UAID_E04F43E67EF6227D01_1896464149", client=client)
    forest_env = ForestEnvPCGClass("PCGDemo_ForestBP_C_UAID_E04F43E67EF62B7D01_2006191743", client)

    # City
    police_car = BaseObjectClass(["BP_vehCar_vehicle13_C_UAID_08BFB8191750913902_2094620511"], client=client)
    beige_sport_car = BaseObjectClass(["BP_vehCar_vehicle13_C_UAID_08BFB8191750913902_2094620511"], client=client)
    blue_sport_car = BaseObjectClass(["BP_vehCar_vehicle06_C_UAID_08BFB8191750B53A02_2077484681"], client=client)
    red_sport_car = BaseObjectClass(["BP_vehCar_vehicle06_C_UAID_08BFB8191750B53A02_2072369680"], client=client)
    white_sport_car = BaseObjectClass(["BP_vehCar_vehicle06_C_UAID_08BFB8191750B53A02_2085692682"], client=client)
    construction_works = BaseObjectClass(["BPP_Park_debris_A_C_UAID_08BFB8191750C63E02_2068574823",
                                          "BPP_Park_debris_B_C_UAID_08BFB8191750C63E02_2066505822",
                                          "BPP_Park_debris_C_C_UAID_08BFB8191750C63E02_2064132821",
                                          "BPP_Park_debris_D_C_UAID_08BFB8191750C63E02_1941156820"], client=client)
    city_fire = BaseObjectClass(["BP_Configuration_13_C_UAID_08BFB8191750E53702_1640163914",
                                 "StaticMeshActor_UAID_08BFB8191750E53702_1100736906",
                                 "BP_Configuration_12_C_UAID_08BFB8191750E53702_1361488908",
                                 "SkeletalMeshActor_UAID_08BFB8191750E43702_1589650724"], client=client)

    black_pickup_truck = BaseObjectClass(["BP_vehTruck_Vehicle04_C_UAID_08BFB8191750B53A02_2047296678",
                                          "BP_vehVan_vehicle09_C_UAID_08BFB8191750B53A02_1996565676"], client=client)

    green_pickup_truck = BaseObjectClass(["BP_vehTruck_Vehicle04_C_UAID_08BFB8191750B53A02_2057827679"], client=client)
    red_pickup_truck = BaseObjectClass(["BP_vehTruck_Vehicle04_C_UAID_08BFB8191750B53A02_2034821677"], client=client)
    white_pickup_truck = BaseObjectClass(["BP_vehTruck_Vehicle04_C_UAID_08BFB8191750913902_1558904503",
                                          "BP_vehVan_vehicle09_C_UAID_08BFB8191750913902_2076347509"], client=client)

    crowd = PCGClass("UNKNOWN", "UNKNOWN", client=client)
    city_trash = PCGClass("UNKNOWN", "UNKNOWN", client=client)
    anomaly = BaseObjectClass([
        "Actor_UAID_08BFB8191750653A02_1780219521",  # Animal
        "SkeletalMeshActor_UAID_08BFB8191750B53A02_1807591675",  # Dino
        "Actor_UAID_08BFB8191750653A02_1976051522",  # Helicopter
        "Actor_UAID_08BFB8191750653A02_1345980520",  # Plane
        "SkeletalMeshActor_UAID_08BFB8191750B63A02_1935932861",  # Tank
        "Actor_UAID_08BFB8191750653A02_1095763519",  # Ufo
    ], client=client)

    city_env = PCGClass("UNKNOWN", ["UNKNOWN"],
                        client=client)  # Yes, city_env actually moves the spawner so it's PCGClass not PCGEnvClass

    return {
        ForestScenarioMapper.ObjectType.FIRE: forest_fire,
        ForestScenarioMapper.ObjectType.TRASH: forest_trash,
        ForestScenarioMapper.ObjectType.CAMPING: camping,
        ForestScenarioMapper.ObjectType.BUILDING: building,
        ForestScenarioMapper.ObjectType.PERSON: person,
        ForestScenarioMapper.ObjectType.UFO: ufo,
        ForestScenarioMapper.ObjectType.PLANE: plane,
        ForestScenarioMapper.ObjectType.HELICOPTER: helicopter,
        "SUN": sun,
        "FOREST": forest_env,

        CityScenarioMapper.ObjectType.POLICE_CAR: police_car,
        CityScenarioMapper.ObjectType.BEIGE_SPORT_CAR: beige_sport_car,
        CityScenarioMapper.ObjectType.BLUE_SPORT_CAR: blue_sport_car,
        CityScenarioMapper.ObjectType.RED_SPORT_CAR: red_sport_car,
        CityScenarioMapper.ObjectType.WHITE_SPORT_CAR: white_sport_car,
        CityScenarioMapper.ObjectType.CONSTRUCTION_WORKS: construction_works,
        CityScenarioMapper.ObjectType.FIRE: city_fire,
        CityScenarioMapper.ObjectType.BLACK_PICKUP_TRUCK: black_pickup_truck,
        CityScenarioMapper.ObjectType.GREEN_PICKUP_TRUCK: green_pickup_truck,
        CityScenarioMapper.ObjectType.RED_PICKUP_TRUCK: red_pickup_truck,
        CityScenarioMapper.ObjectType.WHITE_PICKUP_TRUCK: white_pickup_truck,
        CityScenarioMapper.ObjectType.CROWD: crowd,
        CityScenarioMapper.ObjectType.TRASH: city_trash,
        CityScenarioMapper.ObjectType.ANOMALY: anomaly,
        "CITY": city_env,
    }
