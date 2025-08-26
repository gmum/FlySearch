# THOSE ARE EXAMPLES. DO NOT RUN THEM DIRECTLY. INSTEAD, COPY THE APPROPRIATE EXAMPLE TO FlySearch/script.sh AND RUN IT (AFTER MODIFYING IT AS NEEDED).

# This script creates two FS-1-like scenarios  and then tests gemini-2.5-flash on them.
python3 drone.py \
--scenario_type city_random \
--height_min 30 \
--height_max 100 \
--model "gemini-2.5-flash" \
--log_directory "all_logs" \
--run_name "GeminiFlash25-FS1-Gen-EX1" \
--dummy_first "true" \
--forgiveness 5 \
--glimpses 10 \
--number_of_runs 2 \
--agent "simple_llm" \
--line_of_sight_assured "true" \
--alpha 0.5 \
--show_class_image "false" \
--prompt_type "fs1" \
--random_sun "false"

# # This script creates five FS-2-like scenarios and then tests gemini-2.5-flash on them.
python3 drone.py \
--scenario_type city_random \
--height_min 100 \
--height_max 125 \
--model "gemini-flash-2.5" \
--log_directory "all_logs" \
--run_name "GeminiFlash25-FS2-Gen-EX1" \
--dummy_first "true" \
--forgiveness 5 \
--glimpses 20 \
--number_of_runs 5 \
--agent "simple_llm" \
--line_of_sight_assured "false" \
--alpha 0.95 \
--show_class_image "true" \
--prompt_type "fs2" \
--random_sun "true"