#python3 drone.py --prompt xml_grid --glimpses 50 --glimpse_generator grid --model gpt-4o --run_name newlogs4-big3 --repeats 5 --response_parser xml
python3 drone.py \
--prompt proximity \
--glimpses 5 \
--glimpse_generator grid \
--model intern \
--run_name scenarios_nice_1 \
--scenario_type level_1_yellow_truck \
--response_parser xml \
--height_min 20 \
--height_max 50 \
--height_step 10 \
--x_offset_min -20 \
--x_offset_max 20 \
--x_offset_step 10 \
--y_offset_min -20 \
--y_offset_max 20 \
--y_offset_step 10 \
-- logdir ../plgwtln/results


