# micro-cosmos

It's a micro enviroment simulation, where NEAT powered bugs try to survive searching for food and fighting each other.
![example](https://github.com/HVisMyLife/micro-cosmos/blob/master/assets/docs/screen.png)

It uses bevy and it's entity component system, instead of, more traditional object oriented approach.
Every bug have it's own independently evolving neural network and cone-shaped field of view. 
NN inputs:
 - self velocity;
 - self angular velocity;
 - hp amount;
 - hunger amount;
 - distance to closest bug in sight
 - relative angle to closest bug in sight
 - distance to closest food in sight
 - relative angle to closest food in sight

When bug runs out of hunger, it starts to loose health.
Eating food replenishes hunger bar, eating other bugs replenishes food bar and health bar (at customizable rates).
When hunger bar is full bug spawns offspring with one random mutation in it's genome at the cost of part of it's hunger.
At the start there is loads of food, as time passes less and less eaten food is being respawned.
Varying world parameters allows to modify bugs behaviour towards for example more hostility to each other.

I'm planning to add:
 - species and cross-breeding (my neat lib needs to be expanded);
 - more hostile bugs gain more from eating other bugs, etc;
 - bugs can choose to not eat others/food even when in contact with them (rewarding being in packs);
