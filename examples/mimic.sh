# THOSE ARE EXAMPLES. DO NOT RUN THEM DIRECTLY. INSTEAD, COPY THE APPROPRIATE EXAMPLE TO FlySearch/script.sh AND RUN IT (AFTER MODIFYING IT AS NEEDED).

# This script tests gpt-4o on the FS-1 and FS-Anomaly-1 at the same time (city environment); scenarios 1-200 are FS-1 and scenarios 201-300 are FS-Anomaly-1.
python3 drone.py \
--scenario_type mimic \
--mimic_run_path "run_templates/city-template" \
--mimic_run_cls_names "*" \
--model "gpt-4o" \
--log_directory "all_logs" \
--run_name "GPT4o-FS1(A)-city-EXAMPLE" \
--dummy_first "true" \
--forgiveness 5 \
--glimpses 10 \
--number_of_runs 300 \
--agent "simple_llm" \
--show_class_image "false" \
--prompt_type "fs1"

# This script tests Gemma27b on the FS2 scenario and logs the results to all_logs/Gemma27b-FS-2-EXAMPLE.
python3 drone.py \
--scenario_type mimic \
--mimic_run_path "run_templates/fs2-template" \
--mimic_run_cls_names "*" \
--model "google/gemma-3-27b-it" \
--log_directory "all_logs" \
--run_name "Gemma27b-FS2-EXAMPLE" \
--dummy_first "true" \
--forgiveness 5 \
--glimpses 20 \
--number_of_runs 200 \
--agent "simple_llm" \
--show_class_image "true" \
--prompt_type "fs2"