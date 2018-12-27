##**StarCraft Bots**
Scripts which create an interface for learning algorithms to interact with Sc2 and then generate and train those learning algorithms.  I am not a prominent starcraft player.
###Modules
SC
random
cv2
numpy
time
keras
math
os
### Description
Contains two files, each of which are based around a central script which provides a intermediate layer between a learning algorithm and the game itself. The intermediate script converts numerical inputs into actions, and processes the gamestate to return a pixel map which a learning algorithm may take as input. Marie proved capable of using a learning military model alongside a non learning economic model to defeat all the standard in game AI. Osal is currently experiencing issues with feedback loops and tends to default towards inneffective strategies.
### Marie
Marie is named in the style of Human AI within the Halo Franchise. Marie is focused on using the Terran faction.  Rather than giving a single algorithm complete control, Marie is structured such that different algorithms are responsible for the military and economic aspects of the game, allowing them to hopefully be trained individually, such that the changes made in a single area are more clear.
### Osal
Osal, Obedient Salvation, is named in the style of Forerunner AI in the Halo franchise. Osal gives complete control to a single algorithm, allowing it to better focus on a single aspect.
