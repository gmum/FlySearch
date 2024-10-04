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
