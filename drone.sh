export CITY_BINARY_PATH="/home/dominik/MyStuff/simulator/CitySample/Binaries/Linux/CitySample"
export FOREST_BINARY_PATH="/home/dominik/MyStuff/simulator-dreamsenv/Linux/ElectricDreamsEnv/Binaries/Linux/ElectricDreamsSample"
export LOCATIONS_CITY_PATH="/home/dominik/MyStuff/active-visual-gpt/locations_city.csv"
export FONT_LOCATION="/usr/share/fonts/google-noto/NotoSerif-Bold.ttf"

python3 drone.py \
--prompt xml_grid_grid_found \
--glimpses 5 \
--glimpse_generator grid \
--model gpt-4o \
--run_name A8 \
--scenario_type city \
--navigator grid \
--height_min 30 \
--height_max 100 \
--incontext False \
--repeats 1 \
--logdir all_logs \
--n 1


