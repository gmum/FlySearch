export CITY_BINARY_PATH="/home/dominik/MyStuff/simulator/simulator/CitySample/Binaries/Linux/CitySample"
export FOREST_BINARY_PATH="/home/dominik/MyStuff/simulator-dreamsenv/Linux/ElectricDreamsEnv/Binaries/Linux/ElectricDreamsSample"
export LOCATIONS_CITY_PATH="/home/dominik/MyStuff/active-visual-gpt/locations_city.csv"
export FONT_LOCATION="/usr/share/fonts/google-noto/NotoSerif-Bold.ttf"

#python3 drone.py \
#--scenario_type mimic \
#--mimic_run_path run_templates/a20 \
#--mimic_run_cls_names "*" \
#--continue_from 0 \
#--model gpt-4o \
#--log_on_wandb True \
#--wandb_project_name "WTLN-RF-1" \
#--log_directory "all_logs" \
#--run_name "WTLN-A20-2" \
#--dummy_first False \
#--forgiveness 5 \
#--glimpses 10 \
#--number_of_runs 20 \
#--agent "description_llm"

#python3 drone.py \
#--scenario_type city_random \
#--height_min 100 \
#--height_max 125 \
#--model gpt-4o \
#--log_on_wandb True \
#--wandb_project_name "WTLN-RF-1" \
#--log_directory "all_logs" \
#--run_name "GPT4o-FS2SUN2BOUNDSFIXEDMIMIC" \
#--dummy_first False \
#--forgiveness 5 \
#--glimpses 20 \
#--number_of_runs 200 \
#--agent "simple_llm" \
#--line_of_sight_assured "false" \
#--alpha 0.95 \
#--show_class_image "true" \
#--prompt_type "fs2" \
#--random_sun "true"

python3 drone.py \
--scenario_type mimic \
--mimic_run_path all_logs/GPT4o-FS2SUN1 \
--mimic_run_cls_names "*" \
--model gpt-4o \
--log_on_wandb True \
--wandb_project_name "WTLN-RF-1" \
--log_directory "all_logs" \
--run_name "GPT4o-FS2SUN2BOUNDSFIXEDMIMIC" \
--dummy_first False \
--forgiveness 5 \
--glimpses 20 \
--number_of_runs 200 \
--agent "simple_llm" \
--line_of_sight_assured "false" \
--show_class_image "true" \
--prompt_type "fs2"