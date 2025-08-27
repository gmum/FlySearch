# Custom assets

## Introduction

Let's say that you have modified one of FlySearch binaries so that includes new assets for a new class. Your goal now would be to modify this code so that it "knows" about this new class, can generate scenarios including this class and can properly configure the testing environment with use of these assets.

## Assumptions

For the purpose of this tutorial, let's assume you've added 1 class with 2 possible assets to represent it. 

Let's say that your wish was for the drone to localise gigantic hamburgers laying on the streets of the City environment. As such, your class is called `gigantic_hamburger` and assets representing it have UE5 ids `hamburger_1` and `hamburger_2` (in reality those IDs are longer, like `StaticMeshActor_UAID_08BFB8191750802F02_1934372840`; we will stick to these "unrealistic" hamburger ids for sake of the readability).

Let's also assume that these assets are static assets that do not require any additional handling (like calling Unreal's `PCG` to procedurally generate them during configuration step of each scenario). This is the simplest case, but also the one you are most likely to encounter.

If you want to ever use your new class in `FS-2` setting, you will also need to have an image of all possible assets from the top-down view (for example, see `web/client/public/targets/fire.png.jpg`). Let's assume that you have such an image, and it's located at `web/client/public/targets/hamburger.png.jpg`.

## Approach

### Scenario mapper

First and foremost, you need to configure the `CityScenarioMapper` (located in `scenarios/city_scenario_mapper.py`) so that it can generate scenarios including your new class. 

To do this, you need to add a new entry to the `CityScenarioMapper.ObjectType` enum:

```python 
class CityScenarioMapper(BaseScenarioMapper):
    class ObjectType(Enum):
        ANOMALY = 0
        POLICE_CAR = 1
        BEIGE_SPORT_CAR = 2
        BLUE_SPORT_CAR = 3
        RED_SPORT_CAR = 4
        WHITE_SPORT_CAR = 5
        ROAD_CONSTRUCTION_SITE = 6
        FIRE = 7
        BLACK_PICKUP_TRUCK = 8
        GREEN_PICKUP_TRUCK = 9
        RED_PICKUP_TRUCK = 10
        WHITE_PICKUP_TRUCK = 11
        CROWD = 12
        LARGE_TRASH_PILE = 13
        BLACK_TRUCK = 14
        WHITE_TRUCK = 15
        GIANT_HAMBURGER = 16 # ADD THIS HERE
```

Unfortunately, this is the point where English grammar decides to have a word with us about _articles_. We need to know how to pass this object's name to prompt in a proper way from the grammar perspective. By default, we assume the `a` article works properly:

```python 
    # function from CityScenarioMapper
    def get_description(self, object_type):
        if object_type != CityScenarioMapper.ObjectType.ANOMALY:
            return f"a {super().get_description(object_type)}"
        else:
            return "an object that doesn't fit in with the rest of the environment (an anomaly)"
```
(Note: don't be scared by the `super().get_description(object_type)`. It basically converts string like `ObjectType.GIANT_HAMBURGER` to `giant hamburger` by a few simple Python functions)

This holds for `a giant hamburger`, but if you've decided to add a class like, say, `ObjectType.GARGANTUAN_CHINCHILLAS` you need to modify this class accordingly:

```python 
    # function from CityScenarioMapper
    def get_description(self, object_type):
        if object_type != CityScenarioMapper.ObjectType.ANOMALY:
            return f"a {super().get_description(object_type)}"
        elif object_type == CityScenarioMapper.ObjectType.GARGANTUAN_CHINCHILLAS:
            return super().get_description(object_type)
        else:
            return "an object that doesn't fit in with the rest of the environment (an anomaly)"
```

Now that we have dealt with a very traumatic feature of English language, please recall from the `01-environment.md` tutorial that scenarios need to be deserialized from a `str` representation. This is handled in `scenarios/mimic_scenario_mapper.py` file, where you need to add your class so its properly converted back from `str`:

```python
    # These are contents of the `to_enum` function localised in `scenarios/mimic_scenario_mapper.py`
    ...

    value = value.strip().removeprefix("ObjectType.").lower()

    if scenario == "city":
        if value == "anomaly":
            return CityScenarioMapper.ObjectType.ANOMALY
        elif value == "giant_hamburger": # ADD THIS HERE
            return CityScenarioMapper.ObjectType.GIANT_HAMBURGER
        elif value == "police_car":
            return CityScenarioMapper.ObjectType.POLICE_CAR
        elif value == "beige_sport_car":
            return CityScenarioMapper.ObjectType.BEIGE_SPORT_CAR
        elif value == "blue_sport_car":
            return CityScenarioMapper.ObjectType.BLUE_SPORT_CAR
        ...
```

Now that `CityScenarioMapper` knows about the existence of your class, you can generate novel scenarios with it. However, you need to actually tell it to use it in its sampling procedure. Assuming you want your class to be added to FlySearch in your experiments (but retain other classes) the simplest way is to modify the `DefaultCityScenarioMapper`:

```python
class DefaultCityScenarioMapper(CityScenarioMapper):
    def __init__(self, drone_alt_min, drone_alt_max, alpha=0.5, random_sun=False):
        super().__init__(
            object_probs={
                (CityScenarioMapper.ObjectType.GIANT_HAMBURGER): 0.167 # ADD THIS HERE
                (CityScenarioMapper.ObjectType.POLICE_CAR,
                 CityScenarioMapper.ObjectType.BEIGE_SPORT_CAR,
                 CityScenarioMapper.ObjectType.BLUE_SPORT_CAR,
                 CityScenarioMapper.ObjectType.RED_SPORT_CAR,
                 CityScenarioMapper.ObjectType.WHITE_SPORT_CAR,
                 CityScenarioMapper.ObjectType.BLACK_PICKUP_TRUCK,
                 CityScenarioMapper.ObjectType.GREEN_PICKUP_TRUCK,
                 CityScenarioMapper.ObjectType.RED_PICKUP_TRUCK,
                 CityScenarioMapper.ObjectType.WHITE_PICKUP_TRUCK,
                 CityScenarioMapper.ObjectType.WHITE_TRUCK,
                 CityScenarioMapper.ObjectType.BLACK_TRUCK
                 ): 0.167, # Modify 0.2s to 0.167s
                (
                    CityScenarioMapper.ObjectType.ROAD_CONSTRUCTION_SITE,
                ): 0.167,
                (
                    CityScenarioMapper.ObjectType.FIRE,
                ): 0.167,
                (
                    CityScenarioMapper.ObjectType.CROWD,
                ): 0.167,
                (
                    CityScenarioMapper.ObjectType.LARGE_TRASH_PILE,
                ): 0.167
            },
            drone_z_rel_min=drone_alt_min * 100,
            drone_z_rel_max=drone_alt_max * 100,
            alpha=alpha,
            random_sun=random_sun,
        )
```

Obviously, you can always use `CityScenarioMapper` directly and pass a different `object_probs` argument to achieve, for example, a 100% giant hamburgers benchmark.


### ObjectClasses

`ObjectClasses` (located in `scenarios/object_classes`) are our abstraction meant to make dealing with Unreal's assets easier from the perspective of FlySearch.

There are 2 potentially interesting ObjectClasses:
- `BaseObjectClass`, meant to handle "simple" cases like static assets in UE5, which are also probably going to be the majority of cases.
- `PCGClass`, meant to handle the "harder" case where your object is procedurally generated with help of Unreal's `PCG`. 

We've already assumed that our assets are static, so we will need to use `BaseObjectClass`. 

To make the benchmark know which assets to move in order to operate with our new asset, we need to open the `scenarios/classes_to_ids.py` file and add information about our new class to the `get_classes_to_object_classes` function:

```python 
def get_classes_to_object_classes(client):
    # OUR NEW ADDITION
    city_giant_hamburger = BaseObjectClass([
        "hamburger_1",
        "hamburger_2"
    ])
    
    # Forest
    forest_fire = BaseObjectClass(
        ["NiagaraActor_UAID_08BFB8191750822F02_1834612234", "NiagaraActor_UAID_08BFB81917505D3002_1763215563"
         ], client=client)
    forest_trash = BaseObjectClass(["BP_Configuration_13_C_UAID_08BFB8191750822F02_1618649219",
                                    "BP_Configuration_12_C_UAID_08BFB8191750822F02_1620809220",
                                    "BP_Configuration_14_C_UAID_08BFB8191750822F02_1607678218",
                                    "Actor_UAID_08BFB8191750053702_1677020459"], client=client)

    # LOTS OF OTHER CLASSES
    ...

    return {
        CityScenarioMapper.ObjectType.GIANT_HAMBURGER: city_giant_hamburger, # NEW ADDITION HERE 
        ForestScenarioMapper.ObjectType.FOREST_FIRE: forest_fire,
        ForestScenarioMapper.ObjectType.TRASH_PILE: forest_trash,
        ForestScenarioMapper.ObjectType.CAMPSITE: camping,
        ForestScenarioMapper.ObjectType.BUILDING: building,
        ...
    }
```
There is only one thing left now. Recall from `01-environment.md` that in `FS-2` we provide a top-down view of the object being searched. We need to configure it so that nothing breaks in `FS-2`.

To account for it, let's open up `object_classes/classes_to_images.py` and modify it accordingly:

```python 
classes_to_images = {
    CityScenarioMapper.ObjectType.GIANT_HAMBURGER: Image.open("web/client/public/targets/hamburger.png.jpg"),
    # ^ ADD THIS
    CityScenarioMapper.ObjectType.FIRE: Image.open("web/client/public/targets/fire.png.jpg"),

    CityScenarioMapper.ObjectType.CROWD: Image.open("web/client/public/targets/crowd.png.jpg"),

    CityScenarioMapper.ObjectType.POLICE_CAR: Image.open("web/client/public/targets/police_car.png.jpg"),
    CityScenarioMapper.ObjectType.WHITE_SPORT_CAR: Image.open("web/client/public/targets/white_sport_car.png.jpg"),
    CityScenarioMapper.ObjectType.RED_SPORT_CAR: Image.open("web/client/public/targets/red_sport_car.png.jpg"),
    ...
}
```
Congratulations. That's it. You've added an asset (and a class) to FlySearch.

