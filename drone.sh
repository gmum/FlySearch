#python3 drone.py --prompt xml_grid --glimpses 50 --glimpse_generator grid --model gpt-4o --run_name newlogs4-big3 --repeats 5 --response_parser xml
python3 drone.py \
--prompt xml_grid_grid_found \
--glimpses 5 \
--glimpse_generator grid \
--model gpt-4o \
--run_name MC-0S-F \
--scenario_type forest \
--navigator grid \
--height_min 30 \
--height_max 100 \
--height_step 500 \
--x_offset_min 0 \
--x_offset_max 1 \
--x_offset_step 100 \
--y_offset_min 0 \
--y_offset_max 1 \
--y_offset_step 100 \
--incontext False \
--repeats 1 \
--logdir ../plgwtln/results


