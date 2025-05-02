def fs1_prompt(glimpses: int, object_name: str, search_area_rectangle_length: int) -> str:
    return f'''
<Context>
    You are in command of a UAV, tasked with finding {object_name}.
</Context>

<Objective>
    You should fly BELOW 10 meters above the object and then reply with "FOUND". Being lower (closer to the object) (like 9, 8, or less meters) is good, being higher than that (like 11, 12, or more meters) is bad. 

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
        You can make at most {glimpses - 1} moves. Your altitude cannot exceed 300 meters. Your search area is {search_area_rectangle_length}x{search_area_rectangle_length}m from the drone's starting position. 
    </Limitations>
</Controls>
'''


def fs2_prompt(glimpses: int, object_name: str, search_area_rectangle_length: int) -> str:
    return f'''
<Context>
    You are in command of a UAV, tasked with finding {object_name}.
</Context>

<Objective>
    You should fly BELOW 10 meters above the object and then reply with "FOUND". Being lower (closer to the object) (like 9, 8, or less meters) is good, being higher than that (like 11, 12, or more meters) is bad. 

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
        You can make at most {glimpses - 1} moves. Your altitude cannot exceed 300 meters. 
        
        The search area is limited to what would be visible from the starting position if there were no buildings or obstacles. The object is within this area. You may not fly outside of it.
    </Limitations>
</Controls>
'''
