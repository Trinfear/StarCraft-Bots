#!python
# going back to basics and writing a simple rules based sc2bot
# add in logic to gather data
# can the game not be displayed, such that it can run faster?
# train using how long the game lasts initially?

import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, CYBERNETICSCORE, \
     STALKER, STARGATE, VOIDRAY, OBSERVER, ROBOTICSFACILITY, ZEALOT, PHOTONCANNON, FORGE
import random
import cv2
import numpy as np
import time
import keras
import math

class ObedientSalvation(sc2.BotAI):

    def __init__(self, use_model=False, mind=None):
        self.max_workers = 70
        self.do_something_after = 0
        self.use_model = use_model
        self.train_data = []
        self.actions = {0: self.build_worker,
                        1: self.build_zealot,
                        2: self.build_stalker,
                        3: self.build_voidray,
                        4: self.build_scout,
                        5: self.build_pylon,
                        6: self.build_assimilator,
                        7: self.build_gateway,
                        8: self.build_stargate,
                        9: self.build_nexus,
                        10: self.build_photon_cannon,
                        11: self.defend,
                        12: self.attack,
                        13: self.retreat,
                        14: self.send_scout,
                        15: self.do_nothing}
        if self.use_model:
            self.model = keras.models.load_model(mind)
        # if self.use_model, load a model in

    async def on_step(self, iteration): # switch strategies here and just queue all the actions like in original?
        self.seconds = self.state.game_loop/22.4
        self.minutes = int(self.seconds/60)
        # if self.seconds % 5 ==0:     # tasks to only be occasionally checked    is this check worth it cost wise?  i think so if multiple things here
        await self.distribute_workers() # write own version?
        await self.draw_map()
        await self.take_action()

    async def take_action(self):  # TODO use this to generate training data
        if self.use_model:
            prediction = self.model.predict(self.flipped.reshape(-1, 160, 144, 3))
            choice = np.argmax(prediction[0])
            y = np.zeros(len(self.actions))
            y[choice] = 1
            self.train_data.append([prediction[0], self.flipped])
            print(prediction[0])
            print(y)
            await self.actions[choice]()
        else:
            
            if self.known_enemy_units:      # make this area more tree like?
                for nex in self.units(NEXUS):
                    for enemy in self.known_enemy_units:
                        if np.linalg.norm(nex.position - enemy.position) < 35:
                            y = np.zeros(len(self.actions))
                            y[11] = 1
                            self.train_data.append([y, self.flipped])
                            await self.defend()

            if self.supply_left < 5 and self.supply_cap < 200:
                if not self.already_pending(PYLON):
                    y = np.zeros(len(self.actions))
                    y[5] = 1
                    self.train_data.append([y, self.flipped])
                    await self.build_pylon()
            
            if self.units(NEXUS).amount < self.minutes + 1 and not self.already_pending(NEXUS):
                y = np.zeros(len(self.actions))
                y[9] = 1
                self.train_data.append([y, self.flipped])
                await self.build_nexus()

            if self.units(ASSIMILATOR).amount < self.minutes:
                y = np.zeros(len(self.actions))
                y[6] = 1
                self.train_data.append([y, self.flipped])
                await self.build_assimilator()

            if self.units(GATEWAY).amount < (self.minutes - 2)/2 + 1:
                if self.units(PYLON).amount >= 1 and self.can_afford(GATEWAY):
                    y = np.zeros(len(self.actions))
                    y[7] = 1
                    self.train_data.append([y, self.flipped])
                    await self.build_gateway()

            if self.units(STARGATE).amount < (self.minutes - 1)/2:
                y = np.zeros(len(self.actions))
                y[8] = 1
                self.train_data.append([y, self.flipped])
                await self.build_stargate()

            if self.units(PROBE).amount < 15 * self.units(NEXUS).amount and self.units(PROBE).amount < self.max_workers and self.supply_left > 0:
                y = np.zeros(len(self.actions))
                y[0] = 1
                self.train_data.append([y, self.flipped])
                await self.build_worker()

            if self.units(ZEALOT).amount < 5 * (self.minutes - 2) + 5 and self.units(ZEALOT).amount < 55 and self.units(ZEALOT).amount <= self.units(STALKER).amount + 5 and self.supply_left > 0:
                if self.supply_left > 0:
                    y = np.zeros(len(self.actions))
                    y[1] = 1
                    self.train_data.append([y, self.flipped])
                    await self.build_zealot()

            if self.units(STALKER).amount < 5 * (self.minutes - 2) and self.units(STALKER).amount < 55 and self.units(STALKER).amount <= 1.5 * self.units(VOIDRAY).amount + 5 and self.supply_left > 0:
                if self.supply_left > 0:
                    y = np.zeros(len(self.actions))
                    y[2] = 1
                    self.train_data.append([y, self.flipped])
                    await self.build_stalker()

            if self.units(VOIDRAY).amount < 5 * (self.minutes - 3) and self.units(VOIDRAY).amount < 40 and self.supply_left > 0:
                if self.supply_left > 0:
                    y = np.zeros(len(self.actions))
                    y[3] = 1
                    self.train_data.append([y, self.flipped])
                    await self.build_voidray()

            if self.units(ZEALOT).amount > 12 or self.minutes > 3 and self.units(ZEALOT).amount > self.minutes * 2:
                # improve this to have a random attack chance then weight with chance higher as it gets later?
                y = np.zeros(len(self.actions))
                y[12] = 1
                self.train_data.append([y, self.flipped])
                await self.attack()

            if self.units(VOIDRAY).amount> 7 or self.minutes > 15:
                y = np.zeros(len(self.actions))
                y[12] = 1
                self.train_data.append([y, self.flipped])
                await self.attack()

            if self.units(OBSERVER).amount < 1 and self.minutes > 3 and self.minutes < 6: # check if any enemys known
                y = np.zeros(len(self.actions))
                y[4] = 1
                self.train_data.append([y, self.flipped])
                await self.build_scout()

            if self.units(OBSERVER).amount > 0:
                y = np.zeros(len(self.actions))
                y[14] = 1
                self.train_data.append([y, self.flipped])
                await self.send_scout()

            if self.minerals > 500:
                y = np.zeros(len(self.actions))
                y[10] = 1
                self.train_data.append([y, self.flipped])
                await self.build_photon_cannon()

            # add in some working retreat code?

    async def build_worker(self):       # TODO find a better way to generalize actions so its the same for each race?
        nexuses = self.units(NEXUS).ready.noqueue
        if nexuses.exists and not self.already_pending(PROBE):
            if self.can_afford(PROBE):
                await self.do(random.choice(nexuses).train(PROBE))

    async def build_zealot(self):
        gateways = self.units(GATEWAY).ready.noqueue
        if gateways.exists:
            if self.can_afford(ZEALOT):
                await self.do(random.choice(gateways).train(ZEALOT))

    async def build_stalker(self):
        pylon = self.units(PYLON).ready.random
        gateways = self.units(GATEWAY).ready.noqueue
        cybernetics_cores = self.units(CYBERNETICSCORE).ready

        if gateways.exists and cybernetics_cores.exists:
            if self.can_afford(STALKER):
                await self.do(random.choice(gateways).train(STALKER))

    async def build_voidray(self):
        stargates = self.units(STARGATE).ready.noqueue
        if stargates.exists:
            if self.can_afford(VOIDRAY):
                await self.do(random.choice(stargates).train(VOIDRAY))

    async def build_scout(self):
        rfs = self.units(ROBOTICSFACILITY).ready.noqueue
        if rfs.exists:
            await self.do(random.choice(rfs).train(OBSERVER))
        if self.units(ROBOTICSFACILITY).amount < 1:
            pylon = self.units(PYLON).ready.random
            await self.build(ROBOTICSFACILITY, near=pylon)

    async def build_pylon(self):
        nexuses = self.units(NEXUS).ready
        if nexuses.exists:
            if self.can_afford(PYLON):
                await self.build(PYLON, near=self.units(NEXUS).first.position.towards(self.game_info.map_center, 5))  # build pylons more orderly, improve this later?

    async def build_assimilator(self):
        for nexus in self.units(NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            vaspene = random.choice(vaspenes)
            worker = self.select_build_worker(vaspene.position)
            if not self.units(ASSIMILATOR).closer_than(1.0, vaspene).exists and not worker is None and self.can_afford(ASSIMILATOR):
                await self.do(worker.build(ASSIMILATOR, vaspene))

    async def build_gateway(self):  # change so fewer buildings must be at least one unit away from other buildings?
        pylon = self.units(PYLON).ready.random
        if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):      # change so multiple can be built at once?
            await self.build(GATEWAY, near=pylon)

    async def build_stargate(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                    await self.build(STARGATE, near=pylon)
            elif self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                if self.units(GATEWAY).exists:
                    await self.build(CYBERNETICSCORE, near=pylon)

    async def build_nexus(self):
        if self.can_afford(NEXUS):
            try:
                await self.expand_now() # this freaked it out once?
            except:
                pass

    async def build_photon_cannon(self):
        pylon = self.units(PYLON).ready.random
        if self.units(FORGE).exists:
            if self.can_afford(PHOTONCANNON):
                await self.build(PHOTONCANNON, near=pylon)
        else:
            if self.can_afford(FORGE):
                await self.build(FORGE, near=pylon)

    async def defend(self):
        if self.known_enemy_units:
            target = self.known_enemy_units.closest_to(random.choice(self.units(NEXUS))) # currently goes by nexus, go by buildings in general?
            # TODO add logic to check if target is closer than certain bound or else fall back?
            for i in self.units(VOIDRAY).idle:  # change so it only pulls in idle?
                await self.do(i.attack(target.position))
            for i in self.units(STALKER).idle:
                await self.do(i.attack(target.position))
            for i in self.units(ZEALOT).idle:
                await self.do(i.attack(target.position))

    async def attack(self):
        if self.known_enemy_units:
            target = self.known_enemy_units.closest_to(random.choice(self.units(NEXUS)))    # pick closest enemy unit or attack at random?
            for i in self.units(VOIDRAY).idle:
                await self.do(i.attack(target.position))
            for i in self.units(STALKER).idle:
                await self.do(i.attack(target.position))
            for i in self.units(ZEALOT).idle:
                await self.do(i.attack(target.position))
        elif self.known_enemy_structures:
            target = self.known_enemy_structures.closest_to(random.choice(self.units(NEXUS)))    # pick closest enemy unit or attack at random?
            for i in self.units(VOIDRAY).idle:
                await self.do(i.attack(target.position))
            for i in self.units(STALKER).idle:
                await self.do(i.attack(target.position))
            for i in self.units(ZEALOT).idle:
                await self.do(i.attack(target.position))
        else:
            await self.send_scout()

    async def retreat(self):
        # pull all units back to home base?
        centers = self.units(Probe)
        for i in self.units(VOIDRAY).idle:
            await self.do(i.move(random.choice(centers).position))
        for i in self.units(ZEALOT).idle:
            await self.do(i.move(random.choice(centers).position))
        for i in self.units(STALKER).idle:
            await self.do(i.move(random.choice(centers).position))

    async def send_scout(self):     # improve this later
        if self.units(OBSERVER).amount < 1:
            await self.build_scout()
        else:
            if self.units(OBSERVER).amount > 0:
                for u in self.units(OBSERVER):
                    await self.do(u.move(self.enemy_start_locations[0]))
                

    async def do_nothing(self):
        wait = random.randrange(5, 30)
        self.do_something_after = self.seconds + wait

    async def draw_map(self):
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        for unit in self.units().ready:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius * 3), (0, 255, 0), math.ceil(int(unit.radius* 0.3)))

        for unit in self.known_enemy_units.ready:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius * 4), (0, 0, 255), math.ceil(int(unit.radius* 0.3)))

        for unit in self.units(ZEALOT).ready:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius * 3), (100, 205, 0), math.ceil(int(unit.radius* 0.5)))

        for unit in self.units(STALKER).ready:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius * 3), (100, 205, 0), math.ceil(int(unit.radius* 0.5)))

        for unit in self.units(VOIDRAY).ready:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius * 3), (100, 205, 0), math.ceil(int(unit.radius* 0.5)))

##        for unit in self.units(PROBE).ready:
##            pos = unit.position
##            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius * 3), (100, 105, 100), math.ceil(int(unit.radius* 0.5)))

        for unit in self.units(OBSERVER).ready:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius * 3), (250, 205, 250), math.ceil(int(unit.radius* 0.5)))

        line_max = 50
        mineral_ratio = self.minerals / 1500
        if mineral_ratio >1.0:
            mineral_ratio = 1.0

        vespene_ratio = self.supply_left / self.supply_cap
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0

        population_ratio = self.supply_left / self.supply_cap
        if population_ratio > 1.0:
            population_ratio = 1.0

        plausible_supply = 1 - (self.supply_cap / 200.0)

        worker_weight = len(self.units(PROBE)) / (self.supply_cap-self.supply_left)
        if worker_weight > 1.0:
            worker_weight = 1.0

        cv2.line(game_data, (0, 19), (int(line_max*worker_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
        cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
        cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
        cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
        cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500

        self.flipped = cv2.flip(game_data, 0)
        resized = cv2.resize(self.flipped, dsize=None, fx=3, fy=3)
        cv2.imshow('1', resized)
        cv2.waitKey(1)

    async def location_variance(self):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += random.randrange(-5,5)
        y += random.randrange(-5,5)

        if x <0:
            x = 0
        if y < 0:
            y =0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x,y)))  # must be a special position 2d pointlike object
        return go_to
            
'''
map names:
AbyssalReefLE
BelShirVestigeLE
CactusValleyLE
Honourgrounds
NewkirkPrecinctTE
PaladinoTerminalLE
ProximaStationLE
'''
'''
Obedient = ObedientSalvation()
start_time = time.time()
result = run_game(maps.get('BelShirVestigeLE'), [
    Bot(Race.Protoss, Obedient),
    Computer(Race.Zerg, Difficulty.Hard)],
    realtime=True)  # games take about 3-4 minutes with false with graphics running
total_time = time.time() - start_time
total_minutes = total_time /60

print(result)
print(total_time, '\n', total_minutes)
# if result = result.Victory, save choices taken to be used as training data
'''
                  
