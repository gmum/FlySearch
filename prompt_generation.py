import string


def get_starting_prompt_for_vstar_explorer(number_glimpses: int) -> str:
    return f"""
I need you to provide an answer for a question about the image you see. However, the image provided will be in large resolution and the question in  act may very likely be about very minor details of the image. In fact, you may not be able to see object in question at all.  

You will be presented with options to answer the question. There is always a correct answer among these.

To make it easier for you, you can ask for specific parts (called glimpses) of the image in larger resolution by specifying (x, y) coordinates of the top-left and bottom-right corners of the rectangle you want to see.

For example, if you want to see the top-left corner of the image, you can specify (0, 0) and (0.25, 0.25) as the corners. Of course, you can also go wild and specify coordinates like (0.13, 0.72) and (0.45, 0.89) to see a different part of the image.

The first coordinate is horizontal, the second one is vertical. For example, to get the bottom-left corner of the image, you can specify (0.0, 0.75) and (0.25, 1). To help you out with coordinates, a grid with coordinates is added to the image. It contains dots annotated with their (x, y) coordinates. Use it to your advantage. REMEMBER THAT COORDINATES ARE PROPORTIONAL TO THE IMAGE WIDTH AND HEIGHT. 

Using the same format, please specify the coordinates of the next rectangle you want to see or choose to answer the question. You also MUST specify your reasoning after each decision, as this is beneficial for LLMs, such as you. Put your reasoning in < and >.

YOUR COMMENTS MUST BE PUT IN < AND >. NOTHING ELSE SHOULD BE IN THESE BRACKETS. DO NOT PUT COORDINATES IN THESE BRACKETS.

You can request at most {number_glimpses - 1} glimpses.

OUTPUT FORMAT: (x1, y1) and (x2, y2) OR ANSWER: (your guess).

To answer, you will copy the entire text of the option you think is correct. Do not copy the letter meant to represent option's position.

Do not copy the "Researcher speaks" or "Model speaks" parts of the text. These are only cosmetic to convey the structure of the conversation.

Example: 

=== Researcher speaks ===
(Image of an airport terminal)
Question: Is the red suitcase on the left or right side of the man with Jagiellonian University t-shirt?
Options:
A. The red suitcase is on the left side of the man.
B. The red suitcase is on the right side of the man.

=== Model speaks ===
<I see many suitcases and people in the image, but they aren't zoomed in enough for me to discern details on them. I'll zoom on the bottom left corner of the image to see all people present.> (0.0, 0.75) and (0.25, 1)

=== Researcher speaks ===
(Glimpse of the bottom left corner of the image)

=== Model speaks ===
<I see only one red suitcase, but there are many men and I'm not sure about details on their T-shirts yet. I'll zoom on the suitcase and its horizontal surroundings to see who's wearing the Jagiellonian University T-shirt.> (0.05, 0.75) and (0.20, 0.85)

=== Researcher speaks ===
(Glimpse of the red suitcase and its surroundings)

=== Model speaks ===
<I have found a man with a logo of the Jagiellonian University on his T-shirt. The red suitcase is on his left side.> ANSWER: The red suitcase is on the left side of the man.

<End of example>

DO NOT EXCEED THE NUMBER OF GLIMPSES WHICH IS EQUAL TO {number_glimpses - 1}. YOU CANNOT REQUEST MORE THAN THAT.

Now, you will be presented with the image and the question that you need to answer. Good luck!

"""


def get_starting_prompt_for_vstar_explorer_xml(number_glimpses: int) -> str:
    return f"""
<Instruction>

<Introduction>
I need you to provide an answer for a question about the image you see. However, the image provided will be in large resolution and the question in  act may very likely be about very minor details of the image. In fact, you may not be able to see object in question at all.  

You will be presented with options to answer the question. There is always a correct answer among these.

To make it easier for you, you can ask for specific parts (called glimpses) of the image in larger resolution by specifying (x, y) coordinates of the top-left and bottom-right corners of the rectangle you want to see.

For example, if you want to see the top-left corner of the image, you can specify (0, 0) and (0.25, 0.25) as the corners. Of course, you can also go wild and specify coordinates like (0.13, 0.72) and (0.45, 0.89) to see a different part of the image.

The first coordinate is horizontal, the second one is vertical. For example, to get the bottom-left corner of the image, you can specify (0.0, 0.75) and (0.25, 1). To help you out with coordinates, a grid with coordinates is added to the image. It contains dots annotated with their (x, y) coordinates. Use it to your advantage. REMEMBER THAT COORDINATES ARE PROPORTIONAL TO THE IMAGE WIDTH AND HEIGHT. 

Using the same format, please specify the coordinates of the next rectangle you want to see or choose to answer the question. You also MUST specify your reasoning after each decision, as this is beneficial for LLMs, such as you. Put your reasoning in < and >.

You can request at most {number_glimpses - 1} glimpses.

To answer, you will copy the entire text of the option you think is correct. Do not copy the letter meant to represent option's position.

Your each response should contain <Comment> tag. It should also contain <Request> tag if you are requesting a glimpse. If you are answering the question, it should contain <Answer> tag. Do not put any other tags in your response. Your coordinates should be in (x1, y1) and (x2, y2) format, brackets included. 

</Introduction>

<Example>

<Researcher>
(Image of an airport terminal)
Question: Is the red suitcase on the left or right side of the man with Jagiellonian University t-shirt?
Options:
A. The red suitcase is on the left side of the man.
B. The red suitcase is on the right side of the man.
</Researcher>

<Model>

<Comment>
I see many suitcases and people in the image, but they aren't zoomed in enough for me to discern details on them. I'll zoom on the bottom left corner of the image to see all people present.
</Comment>

<Request>
(0.0, 0.75) and (0.25, 1)
</Request>

</Model>

<Researcher>
(Glimpse of the bottom left corner of the image)
</Researcher>

<Model>
<Comment>
I see only one red suitcase, but there are many men and I'm not sure about details on their T-shirts yet. I'll zoom on the suitcase and its horizontal surroundings to see who's wearing the Jagiellonian University T-shirt.
</Comment>

<Request>
(0.05, 0.75) and (0.20, 0.85)
</Request>
</Model>

<Researcher>
(Glimpse of the red suitcase and its surroundings)
</Researcher>

<Model>
<Comment>
I have found a man with a logo of the Jagiellonian University on his T-shirt. The red suitcase is on his left side.
</Comment>
<Answer>
The red suitcase is on the left side of the man.
</Answer>
</Model>

</Example>

YOU CAN ONLY REQUEST ONE GLIMPSE PER RESPONSE, AND HAVE {number_glimpses - 1} RESPONSES IN TOTAL. EACH RESPONSE MUST CONTAIN A COMMENT TAG. I AM NOT KIDDING, YOU MUST DO THIS.
</Instruction>
"""


def get_classification_prompt_for_vstar_explorer_xml(question: str, options: list[str]) -> str:
    return f"""
<Question>
Question: {question}
Options:
{"\n".join(letter + ". " + option for letter, option in zip(string.ascii_uppercase, options))}
</Question>
"""


def get_classification_prompt_for_vstar_explorer(question: str, options: list[str]) -> str:
    return f"""
Question: {question}
Options:
{"\n".join(letter + ". " + option for letter, option in zip(string.ascii_uppercase, options))}
"""
