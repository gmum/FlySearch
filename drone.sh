export CITY_BINARY_PATH="/home/dominik/MyStuff/simulator/CitySample/Binaries/Linux/CitySample"
export FOREST_BINARY_PATH="/home/dominik/MyStuff/simulator-dreamsenv/Linux/ElectricDreamsEnv/Binaries/Linux/ElectricDreamsSample"
export LOCATIONS_CITY_PATH="/home/dominik/MyStuff/active-visual-gpt/locations_city.csv"
export FONT_LOCATION="/usr/share/fonts/google-noto/NotoSerif-Bold.ttf"

python3 drone.py \
--scenario_type mimic \
--mimic_run_path run_templates/taxi-curse \
--mimic_run_cls_names "*" \
--continue_from 0 \
--model gpt-4o \
--log_on_wandb True \
--wandb_project_name "WTLN-RF-1" \
--log_directory "all_logs" \
--run_name "WTLN-CLN-10" \
--dummy_first False \
--forgiveness 5 \
--glimpses 10 \
--number_of_runs 1 \
--agent "description_llm"
