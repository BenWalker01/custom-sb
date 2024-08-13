from copy import copy
import neat
import pygame
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import main
from Constants import ACTIVE_RUNWAYS,RADAR_UPDATE_RATE
from globalVars import FIXES
from sfparser import loadRunwayData
from PlaneMode import PlaneMode
import math
from shapely.geometry import Point,Polygon
import util
import time
import pickle
from Trainer_Plane import Plane
import random
from prettytable import PrettyTable


WINDOW_SIZE = (800,800)


ROUTES = [["NOVMA","OCK"],
          ["ODVIK","BIG"],
          ["BRAIN","LAM"],
          ["COWLY","BNN"]]

table = PrettyTable()

# Add column names
table.field_names = ["Latitude", "Longitude", "Altitude", "Speed", "Heading","TargetHeading","Mode"]


class Bot:
    def __init__(self,airport):
        self.airport = airport
        self.planes = []

        self.RMA = [(51.726111111111, -0.54972222222222),
                    (51.655833333333, -0.32583333333333),
                    (51.646111111111, 0.15166666666667),
                    (51.505277777778, 0.055277777777778),
                    (51.330875, 0.034811111111111),
                    (51.305, -0.44722222222222),
                    (51.4775, -0.46138888888889),
                    (51.624755555556, -0.51378083333333),
                    (51.726111111111, -0.54972222222222)]
        self.RMA_POLYGON = Polygon(self.RMA)

        self.simulating = True
        self.start_time = None
        self.seen_planes = 1
        self.active_planes = []


    def train_ai(self,genome,config): # put in main.py??
        print("training")
        coords = [FIXES[route[0]] for route in ROUTES]  # Replace getCoords with your actual function

        # Calculate min and max values
        min_lon = min(coord[0] for coord in coords)
        max_lon = max(coord[0] for coord in coords)
        min_lat = min(coord[1] for coord in coords)
        max_lat = max(coord[1] for coord in coords)

        width = util.haversine(min_lat,min_lon,min_lat,max_lon) / 1.852
        height = util.haversine(min_lat,min_lon,max_lat,min_lon) / 1.852
        num_boxes_lon = int(width / 0.3)
        num_boxes_lat = int(height / 0.3)
        print(num_boxes_lat,num_boxes_lon)

        grid = [[[] for _ in range(num_boxes_lat)] for _ in range(num_boxes_lon)]


        net1 = neat.nn.FeedForwardNetwork.create(genome,config)


        route = random.choice(ROUTES)
        lat,lon = FIXES[route[0]]
        head = util.headingFromTo((lat,lon),FIXES[route[-1]])
        p = Plane("TRN101",1000,8000,head,250,lat,lon,0,PlaneMode.HEADING,route[-1])
        self.active_planes.append(p)
        p.start_distance = abs(util.haversine(lat,lon,self.airport[0],self.airport[1]))/ 1.852

        loop_counter = 0 # each loop is 5s
        print("staring")
        while self.simulating:
            # run the sim and see
            if loop_counter > 16:
                route = random.choice(ROUTES)
                lat,lon = FIXES[route[0]]
                head = util.headingFromTo((lat,lon),FIXES[route[-1]])
                p = Plane("TRN101",1000,8000,head,250,lat,lon,0,PlaneMode.HEADING,route[-1])
                self.active_planes.append(p)
                p.start_distance = abs(util.haversine(lat,lon,self.airport[0],self.airport[1]))/ 1.852
                        
                self.seen_planes += 1
                loop_counter = 0

            for plane in self.active_planes:
                box_lon = int((plane.lon - min_lon) / 0.1)
                box_lat = int((plane.lat - min_lat) / 0.1)

                grid[box_lon][box_lat].append(plane)

            inputs = []
            for row in grid:
                for box in row:
                    inputs.append(1 if len(box) > 0 else 0)

            output = net1.activate(tuple(inputs))

            plane_select = output[:53_165]
            plen = plane_select.index(max(plane_select))
            row = plen // num_boxes_lon
            col = plen % num_boxes_lon

            try:
                plane = random.choice(grid[row][col])
            except IndexError as e:
                plane = None

            given_inst = False
            if plane != None:
                insts = output[-106:]

                heading = insts[:71]
                heading_desc = heading.index(max(heading))
                if plane.targetHeading != heading_desc * 5:
                    given_inst = True
                plane.targetHeading = heading_desc * 5
                
                altitude = insts[71:79]
                altitude_desc = altitude.index(max(altitude))

                if plane.altitude < (altitude_desc * 1000) + 1000:
                    plane.climbed = True
                if plane.targetAltitude != (altitude_desc * 1000) + 1000:
                    given_inst = True
                plane.targetAltitude = (altitude_desc * 1000) + 1000

                if given_inst:
                    plane.instructions += 1

                if given_inst and not self.RMA_POLYGON.contains(Point(plane.lat,plane.lon)):
                        plane.vectored_out_rma = True

                speed = insts[27:104]
                speed_desc = speed.index(max(speed))
                if plane.speed < (speed_desc * 5) + 125:
                    plane.sped_up = True
                plane.targetSpeed = (speed_desc * 5) + 125
                clapp = insts[-2:]
                clapp_desc = clapp.index(max(clapp))

                if clapp_desc == 1: # CL/APP
                    runwayData = loadRunwayData("EGLL")["27R"] # TODO get better
                    plane.clearedILS = runwayData
                    plane.mode = PlaneMode.ILS
                    plane.d_clappd = abs(util.haversine(plane.lat,plane.lon,self.airport[0],self.airport[1]))/1.852

                if plane.mode == PlaneMode.HEADING:
                    if not plane.left_rma and not self.RMA_POLYGON.contains(Point(plane.lat,plane.lon)):
                        plane.left_rma = True

            
            for plane in self.active_planes:
                plane.calculatePosition()
                dist_from_airport = abs(util.haversine(plane.lat,plane.lon,self.airport[0],self.airport[1])) / 1.852
                if plane.maxd != None:
                    plane.maxd = max(plane.maxd,dist_from_airport)
                else:
                    plane.maxd = dist_from_airport
                

            loop_counter += 1
            # os.system("cls" if os.name == "nt" else "clear")
            # print(table)

            if self.seen_planes >= 24:
                self.simulating = False
            
    
            

        self.planes.extend(self.active_planes.copy())
        self.calc_fitness(genome)

    def calc_fitness(self,genome):
        #genome.fitness += 1 # TODO : tune this
        for plane in self.planes:
            if plane.intercept_dist != None:
                if plane.intercept_dist < 8:
                    genome.fitness -= 10
                else:
                    diff = abs(plane.intercept_dist - 12)
                    genome.fitness += calc_score(diff) * 10
                    genome.fitness += round(diff,1) * 5

                if math.isclose(plane.altitude_at_intercept,math.tan(math.radians(3)) * plane.intercept_dist * 6076, abs_tol=500):
                    genome.fitness += 20
                else:
                    genome.fitness -= abs(plane.altitude_at_intercept-math.tan(math.radians(3)) * plane.intercept_dist * 6076 ) / 500

            if plane.left_rma:
                genome.fitness -= 100

            if plane.vectored_out_rma:
                genome.fitness -= 75

            if plane.sped_up:
                genome.fitness -= 5

            if plane.climbed:
                genome.fitness -= 10

            if plane.dist_from_behind != None:
                if plane.dist_from_behind < 2.5:
                    genome.fitness -= 100
                elif math.isclose(plane.dist_from_behind,3,abs_tol=0.2):
                    genome.fitness += 100
                else:
                    genome.fitness += min(1/(plane.dist_from_behind - 3),150)

            if plane.maxd < plane.start_distance:
                genome.fitness += 50

            if plane.d_clappd != None and 10 < plane.d_clappd < 18:
                genome.fitness += 20


            genome.fitness -= plane.close_calls * 25
            genome.fitness += (13 - plane.instructions)* 0.1
            if plane.distance_travelled >= 70:
                genome.fitness -= 50
            else:
                genome.fitness += max(60 - plane.distance_travelled,0)

            


def calc_score(number,mean=12,stddev = 1):
    exponent = -((number - mean) ** 2) / (2 * stddev ** 2)
    score = math.exp(exponent)
    return score
            



def eval_genomes(genomes,config):
    print("Evaluating...")
    for i, (genomes_id1,genome1) in enumerate(genomes):
        genome1.fitness = 0
        bot = Bot((51.477697222222, -0.43554333333333))
        bot.train_ai(genome1,config)
        print(f"GENONE {i + 1} DONE")



def run_neat(config):
    #p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-99")
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))


    winner = p.run(eval_genomes,100)

    with open("best.pickle","wb") as f:
        pickle.dump(winner,f)

def test_ai(net):
    pygame.init()
    window = pygame.display.set_mode(WINDOW_SIZE)
    RMA = [(51.726111111111, -0.54972222222222),
                    (51.655833333333, -0.32583333333333),
                    (51.646111111111, 0.15166666666667),
                    (51.505277777778, 0.055277777777778),
                    (51.330875, 0.034811111111111),
                    (51.305, -0.44722222222222),
                    (51.4775, -0.46138888888889),
                    (51.624755555556, -0.51378083333333),
                    (51.726111111111, -0.54972222222222)]
    given_coord = (51.477697222222, -0.43554333333333) # Replace with your actual coordinate

    # Calculate min and max values
    min_x = given_coord[0] - 1  # 60 nautical miles in latitude
    max_x = given_coord[0] + 1
    min_y = given_coord[1] - 60 / 38.4  # 60 nautical miles in longitude at latitude 51
    max_y = given_coord[1] + 60 / 38.4


    RMA = [(int(800 * (y - min_y) / (max_y - min_y)), 800 - int(800 * (x - min_x) / (max_x - min_x))) for x, y in RMA]
    planes = []
    route = random.choice(ROUTES)
    lat,lon = FIXES[route[0]]
    head = util.headingFromTo((lat,lon),FIXES[route[-1]])
    p = Plane("TRN101",1000,8000,head,250,lat,lon,0,PlaneMode.HEADING,route[-1])
    planes.append(p)

    seen_planes = 1
    loop_counter = 0
    running = True
    print("Running...")
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False



        if loop_counter > 16:
            route = random.choice(ROUTES)
            lat,lon = FIXES[route[0]]
            head = util.headingFromTo((lat,lon),FIXES[route[-1]])
            p = Plane("TRN101",1000,8000,head,250,lat,lon,0,PlaneMode.HEADING,route[-1])
            planes.append(p)
      
            seen_planes += 1
            loop_counter = 0

        for plane in planes:
            dists = [(p,abs(util.haversine(p.lat,p.lon,plane.lat,plane.lon))) for p in planes]
            dists = sorted(dists, key=lambda x: x[-1])
            inputs = [[p.lat,p.lon,p.altitude,p.speed,p.heading] for p in planes]
            inputs = [i for li in inputs for i in li]
            if plane.distance_travelled > 5:
                input_nodes = [-1] * 57

                input_nodes[0] = plane.lat
                input_nodes[1] = plane.lon
                input_nodes[2] = plane.altitude
                input_nodes[3] = plane.speed
                input_nodes[4] = plane.heading
                input_nodes[5] = given_coord[0]
                input_nodes[6] = given_coord[1]
                input_nodes[7:] = inputs[:50]

                input_nodes.extend([-1] *( 57 - len(input_nodes)))

                output = net.activate(tuple(input_nodes)) # pop in the thingys
                

                heading = output[:73]
                heading_desc = heading.index(max(heading))
                plane.targetHeading = heading_desc * 5
                
                altitude = output[73:79]
                altitude_desc = altitude.index(max(altitude))
                plane.targetAltitude = (altitude_desc * 1000) + 1000

                speed = output[79:105]
                speed_desc = speed.index(max(speed))
                plane.targetSpeed = (speed_desc * 5) + 125
                
                clapp = output[-2:]
                clapp_desc = clapp.index(max(clapp))

                if clapp_desc == 1: # CL/APP
                    runwayData = loadRunwayData("EGLL")["27R"] # TODO get better
                    plane.clearedILS = runwayData
                    plane.mode = PlaneMode.ILS
                    plane.d_clappd = abs(util.haversine(plane.lat,plane.lon,given_coord[0],given_coord[1]))/1.852

                landed = False
                distances = [30]
                if math.isclose((util.haversine(plane.lat,plane.lon,given_coord[0],given_coord[1]) / 1.852),0.1,abs_tol=0.1) and plane.altitude < 300 and plane.mode == PlaneMode.ILS:

                    try:
                        planes.pop(planes.index(plane))
                    except Exception as e:
                        print(e)

        window.fill((0, 0, 0))
        for plane in planes:
            plane.calculatePosition()
            plane_coord = (plane.lat, plane.lon)  # Replace with the actual attributes if different
            plane_coord = (int(800 * (plane_coord[1] - min_y) / (max_y - min_y)), 800 - int(800 * (plane_coord[0] - min_x) / (max_x - min_x)))
            pygame.draw.circle(window, (0, 255, 0), plane_coord, 2)  # Draw a green circle with radius 10 for the plane




        pygame.draw.polygon(window,(255,0,0),RMA,1)
        pygame.display.flip()

    pygame.quit()




def test_best_network(config):
    with open("best.pickle","rb") as f:
        winner = pickle.load(f)

    winner_net = neat.nn.FeedForwardNetwork.create(winner,config)
    test_ai(winner_net)





if __name__ == "__main__":
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,"config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    run_neat(config)
    test_best_network(config)