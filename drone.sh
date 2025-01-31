export CITY_BINARY_PATH="/yourpath/simulator/CitySample/Binaries/Linux/CitySample"
export FOREST_BINARY_PATH="/yourpath/simulator-dreamsenv/Linux/ElectricDreamsEnv/Binaries/Linux/ElectricDreamsSample"
export LOCATIONS_CITY_PATH="/yourpath/locations_city.csv"
export FONT_LOCATION="/usr/share/fonts/google-noto/NotoSerif-Bold.ttf"

python3 drone.py \
--prompt xml_grid_grid_found \
--glimpses 10 \
--glimpse_generator grid \
--model anthropic-claude-3-5-sonnet-20241022 \
--run_name SomeRun \
--scenario_type forest \
--navigator grid \
--height_min 30 \
--height_max 100 \
--incontext False \
--repeats 1 \
--logdir all_logs \
--n 200 \
--mimic_run_name Sonnet-ForestNew \
--mimic_run_cls_names "*" \
#--continue_from 217