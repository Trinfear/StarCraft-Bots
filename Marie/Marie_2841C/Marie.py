#!python3
# ai template for terran in sc2
# break this into multiple ai, one that does thing like attack and defend, one for economy and one for building units, train them one at a time?
# once all three are trained, use them to create training data for a master ai?

import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer
from sc2.constants import *  # would trimming this down at all actually be helpful?
import random
import cv2
import numpy as np
import time
import keras
import math

class Marie2841C(sc2.BotAI):
    #                   build/expand        attack/defend/scout         anything else?
    def __init__(self, economy_model=False, military_model=False, economist_name='', commander_name=''):
        self.max_workers = 70
        self.do_something_after = 0
        self.economy_model= economy_model
        self.military_model = military_model
        self.economy_data = []
        self.military_data = []
        self.economist_name = economist_name
        self.commander_name = commander_name
        self.scout_sent_time = 0
        self.actions = {0: self.build_worker,   # break into economic and military? currently first 5 are economic
                        1: self.expand,
                        2: self.build_supply,
                        3: self.build_barracks,
                        4: self.build_refinery,  # is this necessary at all?  upgrade baracks, maybe useful later?
                        5: self.build_marine,
                        6: self.attack,
                        7: self.defend,
                        8: self.retreat,
                        9: self.send_scout,
                        10: self.do_nothing}

        if self.economy_model:
            self.economist = keras.models.load_model(economist_name)

        if self.military_model:
            print('commander ready')
            self.commander = keras.models.load_model(commander_name)

    async def on_step(self, iteration):
        self.seconds = self.state.game_loop/22.4
        self.minutes = int(self.seconds/60)
        await self.distribute_workers()
        await self.draw_map()
        await self.economic_action()
        await self.military_action()

    async def economic_action(self):

        if self.economy_model:
            prediction = self.conomist.predict(self.flipped.reshape(-1, 160, 144, 3))
            choice = np.argmax(prediction)
            await self.actions[choice]()
            y = np.zeros(len(self.actions))
            y[choice]=1
            self.economy_data.append([prediction, self.flipped])
        else:
            if self.supply_left < 5 and self.supply_cap < 200 and not self.already_pending(SUPPLYDEPOT):
                y = np.zeros(len(self.actions))
                y[2] = 1
                self.economy_data.append([y, self.flipped])
                await self.build_supply()
                
            if self.units(COMMANDCENTER).amount < self.minutes + 1 and not self.already_pending(COMMANDCENTER):
                y = np.zeros(len(self.actions))
                y[1] = 1
                self.economy_data.append([y, self.flipped])
                await self.expand()

            if self.units(BARRACKS).amount < self.minutes + 1 and self.can_afford(BARRACKS):
                y = np.zeros(len(self.actions))
                y[3] = 1
                self.economy_data.append([y, self.flipped])
                await self.build_barracks()

            if self.units(SCV).amount < 15 * self.units(COMMANDCENTER).amount and self.units(SCV).amount < self.max_workers:
                y = np.zeros(len(self.actions))
                y[0] = 1
                self.economy_data.append([y, self.flipped])
                await self.build_worker()

            if self.units(MARINE).amount < 10 * self.minutes and self.supply_left > 0:
                y = np.zeros(len(self.actions))
                y[5] = 1
                self.economy_data.append([y, self.flipped])
                await self.build_marine()

    async def military_action(self):

        if self.military_model:
            prediction = self.commander.predict(self.flipped.reshape(-1, 160, 144, 3))
            choice = np.argmax(prediction[0])
            await self.actions[choice]() # convert choice to max
            y = np.zeros(len(self.actions))
            print(prediction)
            for i in range(len(y)):
                y[i] = prediction[0][i]
            y[choice]=1
            print(choice)
            self.military_data.append([y, self.flipped])
            
        else:
            if self.units(MARINE).amount > 6 * self.minutes or self.units(MARINE).amount > 50:
                y = np.zeros(len(self.actions))
                y[6] = 1
                self.military_data.append([y, self.flipped])
                await self.attack()
                
            if self.known_enemy_units:      # O(mn) fix this
                for center in self.units(COMMANDCENTER):
                    for enemy in self.known_enemy_units:
                        if np.linalg.norm(center.position - enemy.position) < 35:
                            y = np.zeros(len(self.actions))
                            y[7] = 1
                            self.military_data.append([y, self.flipped])
                            await self.defend(target=enemy)

            if self.units(MARINE).amount < self.minutes * 3 and self.units(MARINE).amount < 70:
                y = np.zeros(len(self.actions))
                y[8] = 1
                self.military_data.append([y, self.flipped])
                await self.retreat()

            if not self.known_enemy_units and not self.known_enemy_structures and self.seconds > self.scout_sent_time + 60:
                y = np.zeros(len(self.actions))
                y[9] = 1
                self.military_data.append([y, self.flipped])
                await self.send_scout()

    async def build_worker(self):
        bases = self.units(COMMANDCENTER).ready.noqueue
        if bases.exists and self.can_afford(SCV):
            await self.do(random.choice(bases).train(SCV))

    async def expand(self):
        if self.can_afford(COMMANDCENTER):
            try:
                await self.expand_now()
            except:
                pass

    async def build_supply(self):
        centers = self.units(COMMANDCENTER).ready
        if centers.exists:
            if self.can_afford(SUPPLYDEPOT):
                await self.build(SUPPLYDEPOT, near=self.units(COMMANDCENTER).first.position.towards(self.game_info.map_center, 5))

    async def build_barracks(self):
        depot = self.units(SUPPLYDEPOT).ready.exists
        if self.can_afford(BARRACKS) and not self.already_pending(BARRACKS):
            await self.build(BARRACKS, near=random.choice(self.units(COMMANDCENTER)))

    async def build_refinery(self):
        for center in self.units(COMMANDCENTER).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, center)
            vaspene = random.choice(vaspenes)
            worker = self.select_build_worker(vaspene.position)
            if not self.units(REFINERY).closer_than(1.0, vaspene).exists and not worker is None and self.can_afford(REFINERY):
                await self.do(worker.build(REFINERY, vaspene))

    async def build_marine(self):
        barracks = self.units(BARRACKS).ready.noqueue
        if barracks.exists and self.can_afford(MARINE):
            await self.do(random.choice(barracks).train(MARINE))

    async def attack(self):
        if self.known_enemy_units:
            target = random.choice(self.known_enemy_units).position
            for i in self.units(MARINE).idle:
                await self.do(i.attack(target))
                
        elif self.known_enemy_structures:
            target = random.choice(self.known_enemy_structures).position
            for i in self.units(MARINE).idle:
                await self.do(i.attack(target))

        else:
            await self.send_scout()

    async def defend(self, target=None):
        if self.known_enemy_units:
            if not target:
                target = self.known_enemy_units.closest_to(random.choice(self.units(COMMANDCENTER)))
            for i in self.units(MARINE).idle:
                await self.do(i.attack(target.position))

    async def retreat(self):
        centers = self.units(SCV)
        for i in self.units(MARINE).idle:
            await self.do(i.move(random.choice(centers).position))

    async def send_scout(self):
        if self.seconds > self.scout_sent_time + 60:
            scout = random.choice(self.units(SCV))
            self.scout_sent_time = self.seconds
            await self.do(scout.move(self.enemy_start_locations[0]))

    async def do_nothing(self):
        pass

    async def draw_map(self):       # if multiple bots run at once, the drawn window switches back and forth between them
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        for unit in self.units().ready:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius * 2), (0, 255, 0), math.ceil(int(unit.radius* 0.3)))

        for unit in self.known_enemy_units.ready:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius * 2), (0, 0, 255), math.ceil(int(unit.radius* 0.3)))

        for unit in self.units(MARINE).ready:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius * 3), (100, 205, 0), math.ceil(int(unit.radius* 0.5)))

        for unit in self.units(BARRACKS).ready:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius * 2), (0, 205, 100), math.ceil(int(unit.radius* 0.3)))

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

    async def location_variance(self, location):
        x = location[0]
        y = location[1]

        x += random.randrange(-15,15)
        y += random.randrange(-15,15)

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
if __name__ == '__main__':
    Marie = Marie2841C()
    start_time = time.time()
    result = run_game(maps.get('BelShirVestigeLE'), [
            Bot(Race.Terran, Marie),
            Computer(Race.Zerg, Difficulty.Easy)],
            realtime=False)
    total_time = time.time() - start_time
    total_minutes = total_time /60

    print(result)
    print(total_time, '\n', total_minutes)
    if result == Result.Victory:
        if Marie.military_model:
            np.save('mil_train_data/{}.npy'.format(str(int(time.time()))), np.array(Marie.Military_data))
        if Marie.economic_model:
            np.save('eco_train_data/{}.npy'.format(str(int(time.time()))), np.array(Marie.economy_data))

    
    
