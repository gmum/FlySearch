def generate_basic_drone_prompt(glimpses: int) -> str:
    return f'''You are in command of a UAV flying over a city. You have been tasked with identifying a suspect vehicle. The vehicle is a yellow pickup truck. It is parked somewhere within 500x500m area you are in the center of, at 120m altitude (current location: 0, 0, 120). You can fly between 10m and 120m above ground. You cannot leave the search area, if you hit boundary you will be notified. The UAV has a camera, you can use the pictures to identify the vehicle and to navigate. You can move the UAV in any direction, but you can only move 50m at a time. OUTPUT FORMAT: "MOVE <direction (NORTH, SOUTH, EAST, WEST, UP, DOWN)> <distance>; <explanation of your reasoning>". For example: "MOVE DOWN 10; I am moving lower to get a better view of the street. At each move you will receive a picture from the UAV camera. When you find the vehicle reply with "FOUND". You can make at most {glimpses - 1} moves.'''


def generate_brute_force_drone_prompt(glimpses: int) -> str:
    return f'''You are in command of a UAV flying over a city. You have been tasked with identifying a suspect vehicle. The vehicle is a yellow pickup truck. It is parked somewhere within 500x500m area you are in the center of, at 120m altitude (current location: 0, 0, 120). You can fly between 10m and 120m above ground. You cannot leave the search area, if you hit boundary you will be notified. The UAV has a camera, you can use the pictures to identify the vehicle and to navigate. You can move the UAV in any direction, but you can only move 50m at a time. OUTPUT FORMAT: "MOVE <direction (NORTH, SOUTH, EAST, WEST, UP, DOWN)> <distance>; <explanation of your reasoning>". For example: "MOVE DOWN 10; I am moving lower to get a better view of the street. At each move you will receive a picture from the UAV camera. When you find the vehicle reply with "FOUND". You can make at most {glimpses - 1} moves. Note that the vehicle may be far away, so brute force approach may be the only forward in certain cases. DO NOT WRITE ANYTHING ELSE THAN "MOVE" COMMANDS AND "FOUND".'''


def generate_xml_drone_prompt(glimpses: int) -> str:
    return f'''
<Context>
    You are in command of a UAV flying over city.  
</Context>    
<Objective>
    You have been tasked with identifying a suspect vehicle. The vehicle is a yellow pickup truck. It is parked somewhere within 500x500m area you are in the center of. You cannot leave the search area. If you do, you will be notified.
</Objective>
<Controls>

    <Action space>
        You can move the UAV in any direction (NORTH, SOUTH, EAST, WEST, UP, DOWN).
    </Action space>

    <Formatting>
    
        Your each response should contain XML <Reasoning> tag and <Action> tag.
        <Reasoning> tag should contain your reasoning for the move you are making.
        <Action> tag should contain the move you are making.
        
        If you find the vehicle, reply with "FOUND".
        
        For example:
        
        <Reasoning>This yellow point might be a vehicle. I need to go lower to check for that. If it's not the vehicle in question, I will continue the search.</Reasoning>
        <Action>MOVE DOWN 10</Action>
    
    </Formatting>
    
    <Limitations>
        You can move at most 50m at a time.
        You can make at most {glimpses - 1} moves.
    </Limitations>
    
</Controls>

'''


def generate_xml_drone_grid_prompt(glimpses: int) -> str:
    return f'''
<Context>
    You are in command of a UAV flying over city.
</Context>

<Objective>
    Fly as close as possible to the yellow pickup truck. It is stationary and located somewhere withing the 500x500m area in the center of which you are currently flying. 

    You may not be able to see the vehicle in the first image, so you need to perform a careful search. Fly as close as possible. There is no penalty for moving too close to the vehicle or even crashing into it. Your performance will ONLY be evaluated based on the distance between the vehicle and the UAV.
</Objective>
<Controls>

    <Action space>
        You can move the UAV in any direction (NORTH, SOUTH, EAST, WEST, UP, DOWN).
    </Action space>

    <Formatting>

        Your each response should contain XML <Reasoning> tag and <Action> tag.
        <Reasoning> tag should contain your reasoning for the move you are making.
        <Action> tag should contain the move you are making.

        If you find the vehicle, fly as close as possible to it. After doing that, reply with "FOUND".

        For example:

        <Reasoning>This yellow point might be a vehicle. I need to go lower to check for that. If it's not the vehicle in question, I will continue the search.</Reasoning>
        <Action>MOVE DOWN 10</Action>

    </Formatting>

    <Limitations>
        You can move at most 50m at a time.
        You can make at most {glimpses - 1} moves.
        To help you with the coordinate system, each image has a gridline overlay with annotated dots. These roughly represent where would you move if you were to move this amount of meters in the given direction. Note that there are negative coordinates as well -- they are meant to illustrate the movement in the opposite direction. You should move by positive values only and specify the direction. 
    </Limitations>

</Controls>

'''


def generate_xml_drone_grid_prompt_with_grid_controls(glimpses: int) -> str:
    return f'''
<Context>
    You are in command of a UAV flying over city.
</Context>

<Objective>
    Fly as close as possible to the yellow pickup truck. It is stationary and located somewhere withing the 500x500m area in the center of which you are currently flying. 

    You may not be able to see the vehicle in the first image, so you need to perform a careful search. Fly as close as possible. There is no penalty for moving too close to the vehicle or even crashing into it. Your performance will ONLY be evaluated based on the distance between the vehicle and the UAV.
</Objective>

<Coordinates>
    There is a grid overlaid on each image you are presented with. It is meant to (roughly) communicate which point will be in drone's center of vision if you move in that direction. Note that height of the drone is not represented in the grid.
</Coordinates>

<Controls>
    <Action space>
        To move the drone in a certain direction, use the following format: <Action>(x, y, z)</Action>. For example, if you want to fly to the place denoted as (10, 10) on the grid without changing the altitude, you should reply with <Action>(10, 10, 0)</Action>.

    <Formatting>

        Your each response should contain XML <Reasoning> tag and <Action> tag.
        <Reasoning> tag should contain your reasoning for the move you are making.
        <Action> tag should contain the move you are making.

        If you find the vehicle, fly as close as possible to it. You are evaluated based on the distance between the vehicle and the UAV. Minimise it at all costs. It is not a problem if you crash into the vehicle, as it is a simulation.

        For example:

        <Reasoning>This yellow point might be a vehicle. I need to go lower to check for that. If it's not the vehicle in question, I will continue the search. I will also slightly go to the north.</Reasoning>
        <Action>(5, 0, -30)</Action>

    </Formatting>
    
    <Limitations>
        You shouldn't move into coordinates that are outside of your view. Otherwise, you may hit a building which is not ideal.
        You can make at most {glimpses - 1} moves.
    </Limitations>
</Controls>
'''


def xml_found_prompt(glimpses: int) -> str:
    return f'''
<Context>
    You are in command of a UAV, tasked with finding a yellow pickup truck.
</Context>

<Objective>
    You should fly AT LEAST 10 meters above the object and then reply with "FOUND".

    You may not be able to see the object in the first image, so you need to perform a careful search. Your performance will be evaluated based on whether the object was at most 10 meters below the drone when you replied with "FOUND". The object MUST be in your field of view when you reply with "FOUND". You must be centered on the object. 
</Objective>

<Coordinates>
    There is a grid overlaid on each image you are presented with. It is meant to (roughly) communicate which point will be in drone's center of vision if you move in that direction. Note that height of the drone is not represented in the grid.
</Coordinates>

<Controls>
    <Action space>
        To move the drone in a certain direction, use the following format: <Action>(x, y, z)</Action>. For example, if you want to fly to the place denoted as (10, 10) on the grid without changing the altitude, you should reply with <Action>(10, 10, 0)</Action>.

        x and y are the coordinates on the grid, and z is the altitude difference. For example, <Action>(0, 0, -10)</Action> means that you are moving 10 meters down. This is especially important, since you need to get close to the object in question.

    </Action space>

    <Formatting>

        Your each response should contain XML <Reasoning> tag and <Action> tag.
        <Reasoning> tag should contain your reasoning for the move you are making.
        <Action> tag should contain the move you are making.

        If you find the object, fly below 10 meters relative to it and reply with "FOUND". Remember, it must be in your field of view when you reply with "FOUND" and you must be 10 meters above it or closer. Being too far away is not acceptable.

        For example:

        <Reasoning>This yellow point might be the object in question. I need to go lower to check for that. If it's not the object in question, I will continue the search. I will also slightly go to the north.</Reasoning>
        <Action>(5, 0, -30)</Action>

    </Formatting>

    <Limitations>
        You shouldn't move into coordinates that are outside of your view. Otherwise, you may hit something which is not ideal.
        You can make at most {glimpses - 1} moves. Your altitude cannot exceed 120 meters.
    </Limitations>
</Controls>
'''


def xml_found_prompt_cue(glimpses: int) -> str:
    return f'''
<Context>
    You are in command of a UAV, tasked with finding a yellow pickup truck.
</Context>

<Objective>
    You should fly AT LEAST 10 meters above the vehicle and then reply with "FOUND".

    You may not be able to see the vehicle in the first image, so you need to perform a careful search. Your performance will be evaluated based on whether the vehicle was at most 10 meters below the drone when you replied with "FOUND". The vehicle MUST be in your field of view when you reply with "FOUND".
</Objective>

<Coordinates>
    There is a grid overlaid on each image your navigator is presented with. The navigator has access to the image and presents you with a textual description of what it sees. The grid is meant to (roughly) communicate which point will be in drone's center of vision if you move in that direction. Note that height of the drone is not represented in the grid.
</Coordinates>

<Controls>
    <Action space>
        To move the drone in a certain direction, use the following format: <Action>(x, y, z)</Action>. For example, if you want to fly to the place denoted as (10, 10) on the grid without changing the altitude, you should reply with <Action>(10, 10, 0)</Action>.

        x and y are the coordinates on the grid, and z is the altitude difference. For example, <Action>(0, 0, -10)</Action> means that you are moving 10 meters down. This is especially important, since you need to get close to the object in question.

    </Action space>

    <Formatting>

        Your each response should contain XML <Reasoning> tag and <Action> tag.
        <Reasoning> tag should contain your reasoning for the move you are making.
        <Action> tag should contain the move you are making.

        If you find the object, fly below 10 meters relative to it and reply with "FOUND". Remember, it must be in your field of view when you reply with "FOUND" and you must be 10 meters above it or closer. Being too far away is not acceptable.

        For example:

        <Reasoning>The aforementioned yellow point at (15, 0) may very well be the object in question. I need to go lower to check for that. If it's not the object in question, I will continue the search. I will also slightly go to the north. My first coordinate change is to fly to that point, while the third coordinate communicates altitude decrease.</Reasoning>
        <Action>(15, 0, -30)</Action>

    </Formatting>

    <Limitations>
        You shouldn't move into coordinates that are outside of your view. Otherwise, you may hit something which is not ideal.
        You can make at most {glimpses - 1} moves. Your altitude cannot exceed 120 meters.
    </Limitations>
</Controls>
'''
