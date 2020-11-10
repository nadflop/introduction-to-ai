import util
import os, sys
import datetime, time
import argparse

class SokobanState:
    # player: 2-tuple representing player location (coordinates)
    # boxes: list of 2-tuples indicating box locations
    def __init__(self, player, boxes):
        # self.data stores the state
        self.data = tuple([player] + sorted(boxes))
        # below are cache variables to avoid duplicated computation
        self.all_adj_cache = None
        self.adj = {}
        self.dead = None
        self.solved = None
    def __str__(self):
        return 'player: ' + str(self.player()) + ' boxes: ' + str(self.boxes())
    def __eq__(self, other):
        return type(self) == type(other) and self.data == other.data
    def __lt__(self, other):
        return self.data < other.data
    def __hash__(self):
        return hash(self.data)
    # return player location
    def player(self):
        return self.data[0]
    # return boxes locations
    def boxes(self):
        return self.data[1:]
    def is_goal(self, problem):
        if self.solved is None:
            self.solved = all(problem.map[b[0]][b[1]].target for b in self.boxes())
        return self.solved
    def act(self, problem, act):
        if act in self.adj: return self.adj[act]
        else:
            val = problem.valid_move(self,act)
            self.adj[act] = val
            return val
    def deadp(self, problem):
        if self.dead is None:
            for b in self.boxes():
                for item in problem.dead_location:
                    if b == item:
                        self.dead = True
                        break
                if self.dead:
                    break
        if self.dead is None:
            self.dead = False
        return self.dead
    def all_adj(self, problem):
        if self.all_adj_cache is None:
            succ = []
            for move in 'udlr':
                ## box_moved is a TRUE or FALSE
                valid, box_moved, nextS = self.act(problem, move)
                if valid:
                    succ.append((move, nextS, 1))
            self.all_adj_cache = succ
        return self.all_adj_cache

class MapTile:
    def __init__(self, wall=False, floor=False, target=False):
        self.wall = wall
        self.floor = floor
        self.target = target

def parse_move(move):
    if move == 'u': return (-1,0)
    elif move == 'd': return (1,0)
    elif move == 'l': return (0,-1)
    elif move == 'r': return (0,1)
    raise Exception('Invalid move character.')

class DrawObj:
    WALL = '\033[37;47m \033[0m'
    PLAYER = '\033[97;40m@\033[0m'
    BOX_OFF = '\033[30;101mX\033[0m'
    BOX_ON = '\033[30;102mX\033[0m'
    TARGET = '\033[97;40m*\033[0m'
    FLOOR = '\033[30;40m \033[0m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class SokobanProblem(util.SearchProblem):
    # valid sokoban characters
    valid_chars = '#@+$*. '

    def __init__(self, map, dead_detection=False):
        self.map = [[]]
        self.dead_detection = dead_detection
        self.init_player = (0,0)
        self.init_boxes = []
        self.numboxes = 0
        self.targets = []
        self.parse_map(map)
        self.dead_location = []
        if self.dead_detection:
            self.dead_location = self.compDead()

    # parse the input string into game map
    # Wall              #
    # Player            @
    # Player on target  +
    # Box               $
    # Box on target     *
    # Target            .
    # Floor             (space)
    def parse_map(self, input_str):
        coordinates = lambda: (len(self.map)-1, len(self.map[-1])-1)
        for c in input_str:
            if c == '#':
                self.map[-1].append(MapTile(wall=True))
            elif c == ' ':
                self.map[-1].append(MapTile(floor=True))
            elif c == '@':
                self.map[-1].append(MapTile(floor=True))
                self.init_player = coordinates()
            elif c == '+':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.init_player = coordinates()
                self.targets.append(coordinates())
            elif c == '$':
                self.map[-1].append(MapTile(floor=True))
                self.init_boxes.append(coordinates())
            elif c == '*':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.init_boxes.append(coordinates())
                self.targets.append(coordinates())
            elif c == '.':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.targets.append(coordinates())
            elif c == '\n':
                self.map.append([])
        assert len(self.init_boxes) == len(self.targets), 'Number of boxes must match number of targets.'
        self.numboxes = len(self.init_boxes)

    def print_state(self, s):
        for row in range(len(self.map)):
            for col in range(len(self.map[row])):
                target = self.map[row][col].target
                box = (row,col) in s.boxes()
                player = (row,col) == s.player()
                if box and target: print(DrawObj.BOX_ON, end='')
                elif player and target: print(DrawObj.PLAYER, end='')
                elif target: print(DrawObj.TARGET, end='')
                elif box: print(DrawObj.BOX_OFF, end='')
                elif player: print(DrawObj.PLAYER, end='')
                elif self.map[row][col].wall: print(DrawObj.WALL, end='')
                else: print(DrawObj.FLOOR, end='')
            print()

    # decide if a move is valid
    # return: (whether a move is valid, whether a box is moved, the next state)
    def valid_move(self, s, move, p=None):
        if p is None:
            p = s.player()
        # print(p)
        dx,dy = parse_move(move)
        x1 = p[0] + dx
        y1 = p[1] + dy
        x2 = x1 + dx
        y2 = y1 + dy
        # print(x1,y1,x2,y2)
        if self.map[x1][y1].wall:
            return False, False, None
        elif (x1,y1) in s.boxes():
            if self.map[x2][y2].floor and (x2,y2) not in s.boxes():
                return True, True, SokobanState((x1,y1),
                    [b if b != (x1,y1) else (x2,y2) for b in s.boxes()])
            else:
                return False, False, None
        else:
            return True, False, SokobanState((x1,y1), s.boxes())

    def isDead(self, cord, direction):
        upperIndex = cord[0] #upper wall index
        lowerIndex = cord[0] #bottom wall index
        leftIndex = cord[1]
        rightIndex = cord[1]

        if direction is "left":
            while(self.map[upperIndex][cord[1]].wall):
                # F
                #check if the rightside of it is aGoal, return False if that's the case
                if self.map[upperIndex][cord[1]+1].target:
                    return False
                #check if its a corner, break if True
                if self.map[upperIndex][cord[1]].wall and self.map[upperIndex-1][cord[1]].wall and self.map[upperIndex-1][cord[1]+1].wall:
                    break
                #move on
                upperIndex -= 1
                #check if its a floor, break if True
                if not self.map[upperIndex][cord[1]].wall:
                    return False

            while(self.map[lowerIndex][cord[1]].wall):
                # L
                #check if the rightside of it is aGoal
                if self.map[upperIndex][cord[1]+1].target:
                    return False
                #check if its a corner, return True since there's no goal in the path and its bounded
                if self.map[lowerIndex][cord[1]].wall and self.map[lowerIndex+1][cord[1]].wall and self.map[lowerIndex+1][cord[1]+1].wall:
                    return True
                #move on
                lowerIndex += 1
                #check if its a floor, break if True
                if not self.map[lowerIndex][cord[1]].wall:
                    break

        elif direction is "right": #right
            try:
                self.map[upperIndex][cord[1]].wall
            except:
                return False
            else:
                while(self.map[upperIndex][cord[1]].wall):
                    # 7
                    #check if the leftside of it is aGoal
                    if self.map[upperIndex][cord[1]-1].target:
                        return False
                    #check if its a corner, break if True
                    if self.map[upperIndex][cord[1]].wall and self.map[upperIndex-1][cord[1]].wall and self.map[upperIndex-1][cord[1]-1].wall:
                        break
                    upperIndex -= 1
                    #check if its a floor, break if True
                    if not self.map[upperIndex][cord[1]].wall:
                        return False

                while(self.map[lowerIndex][cord[1]].wall):
                    # _/
                    #check if the leftside of it is aGoal
                    if self.map[lowerIndex][cord[1]-1].target:
                        return False
                    #check if its a corner, break if True
                    if self.map[lowerIndex][cord[1]].wall and self.map[lowerIndex+1][cord[1]].wall and self.map[lowerIndex+1][cord[1]-1].wall:
                        return True
                    lowerIndex += 1
                    if not self.map[lowerIndex][cord[1]].wall:
                        break

        elif direction is "up":
            while(self.map[cord[0]][leftIndex].wall):
                # /-
                #check if the downside of it is aGoal, return False if that's the case
                if self.map[cord[0]+1][leftIndex].target:
                    return False
                #check if its a corner, break if True
                if self.map[cord[0]][leftIndex].wall and self.map[cord[0]][leftIndex-1].wall and self.map[cord[0]+1][leftIndex-1].wall:
                    break
                #move on
                leftIndex -= 1
                #check if its a floor, break if True
                if not self.map[cord[0]][leftIndex].wall:
                    return False

            while(self.map[cord[0]][rightIndex].wall):
                # 7
                #check if the downside of it is aGoal
                if self.map[cord[0]+1][rightIndex].target:
                    return False
                #check if its a corner, return True since there's no goal in the path and its bounded
                if self.map[cord[0]][rightIndex].wall and self.map[cord[0]][rightIndex+1].wall and self.map[cord[0]+1][rightIndex+1].wall:
                    return True
                #move on
                rightIndex += 1
                #check if its a floor, break if True
                if not self.map[cord[0]][rightIndex].wall:
                    break

        else: #down
            while(self.map[cord[0]][leftIndex].wall):
                # L
                #check if the upside of it is aGoal
                if self.map[cord[0]-1][leftIndex].target:
                    return False
                #check if its a corner, break if True
                if self.map[cord[0]][leftIndex].wall and self.map[cord[0]][leftIndex-1].wall and self.map[cord[0]-1][leftIndex-1].wall:
                    break
                leftIndex -= 1
                #check if its a floor, break if True
                if not self.map[cord[0]][leftIndex].wall:
                    return False
            try:
                self.map[cord[0]][rightIndex].wall
            except:
                return False
            else:
                while(self.map[cord[0]][rightIndex].wall):
                    # _/
                    #check if the upside of it is aGoal
                    if self.map[cord[0]-1][rightIndex].target:
                        return False
                    #check if its a corner, break if True
                    if self.map[cord[0]][rightIndex].wall and self.map[cord[0]][rightIndex+1].wall and self.map[cord[0]-1][rightIndex+1].wall:
                        return True
                    rightIndex += 1
                    if not self.map[cord[0]][rightIndex].wall:
                        break

        return False

    def compDead(self):
        result = set()
        seeWall = False
        print(len(self.map))
        for row in range(1, len(self.map)-1):
            for col in range(len(self.map[row])):
                if self.map[row][col].wall:
                    seeWall = True
                if seeWall and self.map[row][col].floor and not self.map[row][col].target:
                    try:
                        self.map[row-1][col].wall
                        self.map[row+1][col].wall
                        self.map[row][col-1].wall
                        self.map[row][col+1].wall
                    except:
                        continue
                    else:
                        if (self.map[row-1][col].wall or self.map[row+1][col].wall) and (self.map[row][col-1].wall or self.map[row][col+1].wall):
                            result.add((row,col))
                        #check if its an edge deadpoint
                        elif self.map[row-1][col].wall:
                            if self.isDead((row-1,col), "up"):
                                result.add((row,col))
                        elif self.map[row+1][col].wall:
                            if self.isDead((row+1,col), "down"):
                                result.add((row,col))
                        elif self.map[row][col-1].wall:
                            if self.isDead((row,col-1), "left"):
                                result.add((row,col))
                        elif self.map[row][col+1].wall:
                            if self.isDead((row, col+1), "right"):
                                result.add((row,col))

            seeWall = False

        return list(result)

    def simpleDeadlock(self, s):
        for b in s.boxes():
            #check if there's a box at the right, left, up, down
            b_r = ((b[0],b[1]+1) in s.boxes()) #right
            b_d = ((b[0]+1,b[1]) in s.boxes()) #down
            b_u = ((b[0]-1,b[1]) in s.boxes()) #up
            b_l = ((b[0],b[1]-1) in s.boxes()) #left

            #deadlock detection: two consecutive boxes against a wall
            w_r = self.map[b[0]][b[1]+1].wall
            w_d = self.map[b[0]+1][b[1]].wall
            w_u = self.map[b[0]-1][b[1]].wall
            w_l = self.map[b[0]][b[1]-1].wall

            if (b_d or b_u) and (w_r or w_l):
                return True
            elif (b_r or b_l) and (w_d or w_u):
                return True

        return False
    ##############################################################################
    # Problem 1: Dead end detection                                              #
    # Modify the function below. We are calling the deadp function for the state #
    # so the result can be cached in that state. Feel free to modify any part of #
    # the code or do something different from us.                                #
    # Our solution to this problem affects or adds approximately 50 lines of     #
    # code in the file in total. Your can vary substantially from this.          #
    ##############################################################################
    def dead_end(self, s):
        if not self.dead_detection:
            return False
        return s.deadp(self)

    def start(self):
        return SokobanState(self.init_player, self.init_boxes)

    def goalp(self, s):
        return s.is_goal(self)

    def expand(self, s):
        if self.dead_end(s):
            return []
        return s.all_adj(self)

class SokobanProblemFaster(SokobanProblem):
    ##############################################################################
    # Problem 2: Action compression                                              #
    # Redefine the expand function in the derived class so that it overrides the #
    # previous one. You may need to modify the solve_sokoban function as well to #
    # account for the change in the action sequence returned by the search       #
    # algorithm. Feel free to make any changes anywhere in the code.             #
    # Our solution to this problem affects or adds approximately 80 lines of     #
    # code in the file in total. Your can vary substantially from this.          #
    ##############################################################################
    # I think given the current state, find all ways to move the box to the targets
    # convert player actions to box actions to find all ways
    # Then convert box actions to player actions and return

    def expand(self, s):

        # possible = list()
        teleport = list()
        boxes = s.boxes()

        visited = BFS(self.map, s.player(), s.boxes())
        #print(visited)

        for b in s.boxes():
            for move in 'udlr':
                if move == 'u':
                    player_pos = (b[0] + 1, b[1])
                    pos = (b[0] - 1, b[1])
                elif move == 'd':
                    player_pos = (b[0] - 1, b[1])
                    pos = (b[0] + 1, b[1])
                elif move == 'l':
                    player_pos = (b[0], b[1] + 1)
                    pos = (b[0], b[1] - 1)
                else:
                    player_pos = (b[0], b[1] - 1)
                    pos = (b[0], b[1] + 1)
                # print(move)
                # print(pos)
                if self.dead_detection and pos in self.dead_location:
                    continue
                if not self.map[player_pos[0]][player_pos[1]].wall and player_pos not in boxes:
                        if (self.valid_move(s, move, player_pos)[0]):
                            if player_pos in visited:
                                newBoxes = [bo if bo != b else pos for bo in s.boxes()]
                                newState = SokobanState(b, newBoxes)
                                teleport.append(((move,player_pos),newState, 1))
        #print(teleport)
        return teleport


## Resources:
# https://www.annytab.com/breadth-first-search-algorithm-in-python/
# https://stackoverflow.com/questions/55316735/shortest-path-in-a-grid-using-bfs
def BFS(map, start, boxes):
    # print('BOXES')
    # print(boxes)

    available = list()
    visited = list()

    available.append(start)
    while len(available) > 0:
        curPoint = available.pop(0)
        (row, col) = curPoint
        ## udlr
        if curPoint not in visited:
            visited.append(curPoint)
            adjacent = [(row - 1,col), (row + 1,col), (row,col - 1), (row,col + 1)]
            for nextPoint in adjacent:
                # print("555555555")
                map_value = map[nextPoint[0]][nextPoint[1]]
                # print('HELLO')
                # print(map_value)
                ## Only places a player can stand on
                ## And not in the array of unaccessible already
                if map_value.wall:
                    continue
                elif (nextPoint not in boxes) and (nextPoint not in visited):
                    # print('nextPoint: {}'.format(nextPoint))
                    # print(boxes)
                    available.append(nextPoint)
                    # print('exploring: {}'.format(available))
                    # print('Visited: {}'.format(visited))
    return visited

def BFS2(map, start, end, boxes):

    queue = [(start, [])]
    visited = list()

    while len(queue) > 0:
        curPoint, path = queue.pop(0)
        path.append(curPoint)
        #print("2222222222")
        visited.append(curPoint)

        if curPoint == end:
            return path

        (y,x) = curPoint
        ## udlr
        adjacent = [(y,x + 1), (y,x - 1), (y + 1,x), (y - 1,x)]
        for nextPoint in adjacent:
            #print("555555555")
            map_value = map[nextPoint[0]][nextPoint[1]]
            #print('HELLO')
            #print(map_value)
            ## Only places a player can stand on
            ## And not in the array of unaccessible already
            if map_value.wall:
                continue
            elif nextPoint not in visited and nextPoint not in boxes:
                queue.append((nextPoint, path[:]))
                #print('exploring: {}'.format(queue))
                #print('Visited: {}'.format(visited))
    return []

def BFSmod(map, start, end, boxes):

    movementQueue = []
    queue = [[start]]
    mQueue = [[]]
    explored = []
    mPath = []

    while queue:
        path = queue.pop(0)
        mPath = mQueue.pop(0)
        node = path[-1]
        (y,x) = node
        if node not in explored:
            for move in 'uldr':
                if move == 'u':
                    nextPoint = (y - 1,x)
                elif move == 'd':
                    nextPoint = (y + 1,x)
                elif move == 'l':
                    nextPoint = (y,x - 1)
                elif move == 'r':
                    nextPoint = (y,x + 1)
                map_value = map[nextPoint[0]][nextPoint[1]]
                ## Only places a player can stand on
                ## And not in the array of unaccessible already
                if map_value.wall:
                    continue
                elif nextPoint not in explored and nextPoint not in boxes:
                    newMPath = list(mPath)
                    newMPath.append(move)
                    newPath = list(path)
                    newPath.append(nextPoint)
                    queue.append(newPath)
                    mQueue.append(newMPath)
                    if nextPoint == end:
                        return newMPath
            explored.append(node)
    return []


class Heuristic:
    def __init__(self, problem):
        self.problem = problem

    ##############################################################################
    # Problem 3: Simple admissible heuristic                                     #
    # Implement a simple admissible heuristic function that can be computed      #
    # quickly based on Manhattan distance. Feel free to make any changes         #
    # anywhere in the code.                                                      #
    # Our solution to this problem affects or adds approximately 10 lines of     #
    # code in the file in total. Your can vary substantially from this.          #
    ##############################################################################
    def heuristic(self, s):
        sum_dist = 0
        for i in s.boxes():
            shortest_dist = 2**31
            for j in self.problem.targets:
                temp_dist = abs(i[0] - j[0]) + abs(i[1] - j[1])
                if temp_dist < shortest_dist:
                    shortest_dist = temp_dist
            sum_dist += shortest_dist
        return sum_dist

    ##############################################################################
    # Problem 4: Better heuristic.                                               #
    # Implement a better and possibly more complicated heuristic that need not   #
    # always be admissible, but improves the search on more complicated Sokoban  #
    # levels most of the time. Feel free to make any changes anywhere in the     # # code. Our heuristic does some significant work at problem initialization   #
    # and caches it.                                                             #
    # Our solution to this problem affects or adds approximately 40 lines of     #
    # code in the file in total. Your can vary substantially from this.          #
    ##############################################################################
    def heuristic2(self, s):
        #raise NotImplementedError('Override me')
        sum_dist = 0
        for i in s.boxes():
            shortest_dist = 2**31
            #if there's a deadlock state
            if self.problem.simpleDeadlock(s):
                sum_dist = 2**31
                break
            for j in self.problem.targets:
                temp_dist = abs(i[0] - j[0]) + abs(i[1] - j[1])
                if temp_dist < shortest_dist:
                    shortest_dist = temp_dist
            sum_dist += shortest_dist
        return sum_dist

# solve sokoban map using specified algorithm
def solve_sokoban(map, algorithm='ucs', dead_detection=False):
    # problem algorithm
    if 'f' in algorithm:
        problem = SokobanProblemFaster(map, dead_detection)
    else:
        problem = SokobanProblem(map, dead_detection)
    curPlayer = problem.init_player
    boxes = problem.init_boxes
    # search algorithm
    h = Heuristic(problem).heuristic2 if ('2' in algorithm) else Heuristic(problem).heuristic
    if 'a' in algorithm:
        search = util.AStarSearch(heuristic=h)
    else:
        search = util.UniformCostSearch()

    # solve problem
    search.solve(problem)
    if search.actions is not None:
        print('length {} soln is {}'.format(len(search.actions), search.actions))
    if 'f' in algorithm:
                ### CHANGE THIS
        ### CHANGE THIS
        # print(search.actions)
        allMoves = list()
        actualChar = list()
        moves = []
        path = ""
        for i in search.actions:
            moves = BFSmod(problem.map, curPlayer, i[1], boxes)
            if len(moves) > 0:
                path += ''.join(moves)
            if i[0] == 'u':
                boxes.remove((i[1][0]-1, i[1][1]))
                boxes.append((i[1][0]-2, i[1][1]))
                curPlayer = (i[1][0]-1,i[1][1])
            elif i[0] == 'd':
                boxes.remove((i[1][0]+1, i[1][1]))
                boxes.append((i[1][0]+2, i[1][1]))
                curPlayer = (i[1][0]+1,i[1][1])
            elif i[0] == 'l':
                boxes.remove((i[1][0], i[1][1]-1))
                boxes.append((i[1][0], i[1][1]-2))
                curPlayer = (i[1][0],i[1][1]-1)
            elif i[0] == 'r':
                boxes.remove((i[1][0], i[1][1]+1))
                boxes.append((i[1][0], i[1][1]+2))
                curPlayer = (i[1][0],i[1][1]+1)
            path += i[0]

        #animate_sokoban_solution(problem.map, actualChar)
        return search.totalCost, path, search.numStatesExplored
    else:
        return search.totalCost, search.actions, search.numStatesExplored

# animate the sequence of actions in sokoban map
def animate_sokoban_solution(map, seq, dt=0.2):
    problem = SokobanProblem(map)
    state = problem.start()
    clear = 'cls' if os.name == 'nt' else 'clear'
    for i in range(len(seq)):
        os.system(clear)
        print(seq[:i] + DrawObj.UNDERLINE + seq[i] + DrawObj.END + seq[i+1:])
        problem.print_state(state)
        time.sleep(dt)
        valid, _, state = problem.valid_move(state, seq[i])
        if not valid:
            raise Exception('Cannot move ' + seq[i] + ' in state ' + str(state))
    os.system(clear)
    print(seq)
    problem.print_state(state)

# read level map from file, returns map represented as string
def read_map_from_file(file, level):
    map = ''
    start = False
    found = False
    with open(file, 'r') as f:
        for line in f:
            if line[0] == "'": continue
            if line.strip().lower()[:5] == 'level':
                if start: break
                if line.strip().lower() == 'level ' + level:
                    found = True
                    start = True
                    continue
            if start:
                if line[0] in SokobanProblem.valid_chars:
                    map += line
                else: break
    if not found:
        raise Exception('Level ' + level + ' not found')
    return map.strip('\n')

# extract all levels from file
def extract_levels(file):
    levels = []
    with open(file, 'r') as f:
        for line in f:
            if line.strip().lower()[:5] == 'level':
                levels += [line.strip().lower()[6:]]
    return levels

def solve_map(file, level, algorithm, dead, simulate):
    map = read_map_from_file(file, level)
    print(map)
    tic = datetime.datetime.now()
    cost, sol, nstates = solve_sokoban(map, algorithm, dead)
    toc = datetime.datetime.now()
    print('Time consumed: {:.3f} seconds using {} and exploring {} states'.format(
        (toc - tic).seconds + (toc - tic).microseconds/1e6, algorithm, nstates))
    seq = ''.join(sol)
    print(len(seq), 'moves')
    print(' '.join(seq[i:i+5] for i in range(0, len(seq), 5)))
    if simulate:
        animate_sokoban_solution(map, seq)

def main():
    parser = argparse.ArgumentParser(description="Solve Sokoban map")
    parser.add_argument("level", help="Level name or 'all'")
    parser.add_argument("algorithm", help="ucs | [f][a[2]] | all")
    parser.add_argument("-d", "--dead", help="Turn on dead state detection (default off)", action="store_true")
    parser.add_argument("-s", "--simulate", help="Simulate the solution (default off)", action="store_true")
    parser.add_argument("-f", "--file", help="File name storing the levels (levels.txt default)", default='levels.txt')
    parser.add_argument("-t", "--timeout", help="Seconds to allow (default 300)", type=int, default=300)

    args = parser.parse_args()
    level = args.level
    algorithm = args.algorithm
    dead = args.dead
    simulate = args.simulate
    file = args.file
    maxSeconds = args.timeout

    if (algorithm == 'all' and level == 'all'):
        raise Exception('Cannot do all levels with all algorithms')

    def solve_now(): solve_map(file, level, algorithm, dead, simulate)

    def solve_with_timeout(maxSeconds):
        try:
            util.TimeoutFunction(solve_now, maxSeconds)()
        except KeyboardInterrupt:
            raise
        except MemoryError as e:
            signal.alarm(0)
            gc.collect()
            print('Memory limit exceeded.')
        except util.TimeoutFunctionException as e:
            signal.alarm(0)
            print('Time limit (%s seconds) exceeded.' % maxSeconds)

    if level == 'all':
        levels = extract_levels(file)
        for level in levels:
            print('Starting level {}'.format(level), file=sys.stderr)
            sys.stdout.flush()
            solve_with_timeout(maxSeconds)
    elif algorithm == 'all':
        for algorithm in ['ucs', 'a', 'a2', 'f', 'fa', 'fa2']:
            print('Starting algorithm {}'.format(algorithm), file=sys.stderr)
            sys.stdout.flush()
            solve_with_timeout(maxSeconds)
    else:
        solve_with_timeout(maxSeconds)

if __name__ == '__main__':
    main()
