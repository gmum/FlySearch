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

        If you find the vehicle, fly as close as possible to it. After doing that, reply with "FOUND". You are evaluated based on the distance between the vehicle and the UAV. Minimise it at all costs. It is not a problem if you crash into the vehicle, as it is a simulation.

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