**ADG** means that model receives only textual description of the image.

**EC** means that the model is asked to minimise euclidean distance to the object in question.

**MC** means that the model is asked to fly below 10 meters above the object, have it in its field of view and respond
with "FOUND".

**0S** means 0-shot.

**1S** means 1-shot.

**CT** means starting above the object.

**CR** means starting "from a corner" (i.e. not being directly above the object).

**HM** means that an artificial message from the model is appended to the conversation history that is meant to
encourage it to fly closer to the object. Furthermore, a message reminding the model how to lower the altitude is sent
in that scenario.

**INT** means InternVL-8B quantised model.

**OC** means the old navigation system where drone is (for example) asking us to fly 10 units to the north by typing
`MOVE NORTH 10`.