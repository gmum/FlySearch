from scenarios.city_scenario_mapper import CityScenarioMapper
from scenarios.forest_scenario_mapper import ForestScenarioMapper

classes_to_ids = {
    ForestScenarioMapper.ObjectType.FIRE: "NiagaraActor_UAID_08BFB8191750822F02_1834612234",
    ForestScenarioMapper.ObjectType.TRASH: "Actor_UAID_08BFB8191750053702_1677020459",
    ForestScenarioMapper.ObjectType.CAMPING: "BP_RandomSpawner_C_UAID_08BFB8191750113C02_1871747848",
    ForestScenarioMapper.ObjectType.BUILDING: "StaticMeshActor_UAID_08BFB8191750802F02_1934372840",
    ForestScenarioMapper.ObjectType.PERSON: "SkeletalMeshActor_UAID_08BFB8191750A22F02_1368103644",
    ForestScenarioMapper.ObjectType.UFO: "Actor_UAID_08BFB8191750053702_1321019457",
    ForestScenarioMapper.ObjectType.PLANE: "Actor_UAID_08BFB8191750043702_2140437280",
    ForestScenarioMapper.ObjectType.HELICOPTER: "StaticMeshActor_UAID_08BFB8191750683002_1545951497",
    "SUN": "DirectionalLight_UAID_E04F43E67EF6227D01_1896464149",
    "FOREST": "PCGDemo_ForestBP_C_UAID_E04F43E67EF62B7D01_2006191743",
    CityScenarioMapper.ObjectType.POLICE_CAR: "BP_vehCar_vehicle13_C_UAID_08BFB8191750913902_2094620511",
    CityScenarioMapper.ObjectType.BEIGE_SPORT_CAR: "BP_vehCar_vehicle06_C_UAID_08BFB8191750913902_1546765502",
    CityScenarioMapper.ObjectType.BLUE_SPORT_CAR: "BP_vehCar_vehicle06_C_UAID_08BFB8191750B53A02_2077484681",
    CityScenarioMapper.ObjectType.RED_SPORT_CAR: "BP_vehCar_vehicle06_C_UAID_08BFB8191750B53A02_2072369680",
    CityScenarioMapper.ObjectType.WHITE_SPORT_CAR: "BP_vehCar_vehicle06_C_UAID_08BFB8191750B53A02_2085692682",
    CityScenarioMapper.ObjectType.CONSTRUCTION_WORKS: ["LevelInstance_UAID_08BFB8191750653902_1323601466",
                                                       "LevelInstance_UAID_08BFB8191750653902_1326632467",
                                                       "LevelInstance_UAID_08BFB8191750653902_1329774468",
                                                       "LevelInstance_UAID_08BFB8191750653902_1336069469"],
    CityScenarioMapper.ObjectType.FIRE_CAR: "SkeletalMeshActor_UAID_08BFB8191750E43702_1589650724",
    CityScenarioMapper.ObjectType.FIRE: ["BP_Configuration_13_C_UAID_08BFB8191750E53702_1640163914",
                                         "StaticMeshActor_UAID_08BFB8191750E53702_1100736906",
                                         "BP_Configuration_12_C_UAID_08BFB8191750E53702_1361488908"],
    CityScenarioMapper.ObjectType.BLACK_PICKUP_TRUCK: ["BP_vehTruck_Vehicle04_C_UAID_08BFB8191750B53A02_2047296678",
                                                       "BP_vehVan_vehicle09_C_UAID_08BFB8191750B53A02_1996565676"],
    CityScenarioMapper.ObjectType.GREEN_PICKUP_TRUCK: "BP_vehTruck_Vehicle04_C_UAID_08BFB8191750B53A02_2057827679",
    CityScenarioMapper.ObjectType.RED_PICKUP_TRUCK: "BP_vehTruck_Vehicle04_C_UAID_08BFB8191750B53A02_2034821677",
    CityScenarioMapper.ObjectType.WHITE_PICKUP_TRUCK: ["BP_vehTruck_Vehicle04_C_UAID_08BFB8191750913902_1558904503",
                                                       "BP_vehVan_vehicle09_C_UAID_08BFB8191750913902_2076347509"],
    CityScenarioMapper.ObjectType.CROWD: "BP_RandomCrowd_C_UAID_08BFB81917507B3902_1554570401",
    CityScenarioMapper.ObjectType.TRASH: "BP_RandomSpawner_C_UAID_08BFB81917507F3902_1265735256",
    CityScenarioMapper.ObjectType.ANOMALY: [
        "Actor_UAID_08BFB8191750653A02_1780219521",  # Animal
        "SkeletalMeshActor_UAID_08BFB8191750B53A02_1807591675",  # Dino
        "Actor_UAID_08BFB8191750653A02_1976051522",  # Helicopter
        "Actor_UAID_08BFB8191750653A02_1345980520",  # Plane
        "SkeletalMeshActor_UAID_08BFB8191750B63A02_1935932861",  # Tank
        "Actor_UAID_08BFB8191750653A02_1095763519",  # Ufo
    ],
    "CITY": "BP_MassTrafficParkedVehicleSpawner_C_UAID_E04F43E62770FBF100_1205606496"
}
