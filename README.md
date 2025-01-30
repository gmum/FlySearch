# FlySearch

This is repository containing code for the FlySearch benchmark.

## Entrypoint

We recommend using the `drone.sh` script to run the benchmark. It is a Bash script that calls the `drone.py` script
which performs the entire evaluation. It has several configurable parameters, such as:

## Dependencies

To be able to run the benchmark, you need to download appropriate dependencies and configure some variables. The purpose
of this section is to tell you how to do that.

### Benchmark binaries

Can be downloaded from https://zenodo.org/records/14775310.

`city.tar.gz` contains the city environment and
`forest3.tar.gz` contains the forest environment. Extract them and then modify the `drone.sh` script by:

* setting the `CITY_BINARY_PATH` to `/your_location/simulator/CitySample/Binaries/Linux/CitySample`
* setting the FOREST_BINARY_PATH to
  `your_location/simulator-dreamsenv/Linux/ElectricDreamsEnv/Binaries/Linux/ElectricDreamsSample`

That's all you need to do to configure the binaries!

You can also verify manually that these work on your computer by
running `./simulator/CitySample/Binaries/Linux/CitySample` or
`./simulator-dreamsenv/Linux/ElectricDreamsEnv/Binaries/Linux/ElectricDreamsSample`. These commands should start the UE5
environment and show it on your screen.

### City locations

The file `locations_city.csv` is provided with this repository. Set the `LOCATIONS_CITY_PATH` variable in the `drone.sh`
to location of this file in your filesystem; this file is important for running city scenarios, as it contains
permissible safe locations for objects to be spawned.

### Fonts

The benchmark needs a font to overlay images from the engine with a navigation scaffold. Set the `FONT_LOCATION`
variable in the `drone.sh` script to the location of a font file in your filesystem. The default one is
`/usr/share/fonts/google-noto/NotoSerif-Bold.ttf`, which may or may not be present on your machine.

###                    
