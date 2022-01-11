import re
import numpy as np
import random

SPACE = 0
BLACK = 1 
WHITE = 2
SIZE = 19

def OPPNUM(colornum):
    return colornum%2 + 1

def charchar2pos(s):
    _x = ord(s[0])-97
    _y = ord(s[1])-97
    return _y*19+_x

def pos2charchar(pos):
    _x = pos%19
    _y = int(pos/19)
    
    a = chr(_x+97)
    b = chr(_y+97)
    return a+b

def board2board(vboard):
    board = []
    for ch in vboard.replace('\n', '').replace(' ', ''):
        board.append(int(ch))
    return board

def err(msg):
    print('error:', msg)

class MOVE:
    def __init__(self, color, pos, comment = 'None'):
        self.color = color
        self.pos = pos
        self.comment = comment    

    def GetColor(self):
        return 'B' if self.color == 'B' else 'W'
    
    def GetPositionStr(self):
        alphabet = 'ABCDEFGHJKLMNOPQRSTU'
        return alphabets[ int(self.pos/19) ] + str(self.pos%19 + 1)
    
    def SetComment(self, comment):
        self.comment = comment
    
    def __str__(self):
        return ';{}[{}]'.format(self.color, self.pos)

class BOARD:
    def __init__(self):
        self.FileName = "None"
        self.board = [SPACE] * SIZE * SIZE

    def PutStone(self, move):
        pos = move.pos
        if pos < 0 or pos >= SIZE * SIZE:
            return
        #if(self.board[pos]):
        #    err('Stone exists.' + move.__str__())
        
        self.board[pos] = move.color
        legal_nearby_pos = self.GetLegalNearby(pos)

        for newpos in legal_nearby_pos:
            if self.board[newpos] == OPPNUM(self.board[pos]):
                chi,visited = self.GetBlock(newpos , chi=set(), visited=set())
                if len(chi) == 0:
                    self.ClearStone(visited)
    
    def ClearStone(self, death_stones):
        for deadstone_pos in death_stones:
            self.board[deadstone_pos] = 0        
    
    def GetLegalNearby(self, pos):
        x = pos%19
        y = int(pos/19)
        left = [-1,0]
        right = [1,0]
        up = [0, -1]
        down = [0, 1]
        directions = [left, right, up , down]
        if x == 0:
            directions.remove(left)
        if x == 18:
            directions.remove(right)
        if y == 0:
            directions.remove(up)
        if y == 18:
            directions.remove(down)        

        legal_nearby_pos = []
        
        for direction in directions:
            _x = x + direction[0]
            _y = y + direction[1]
            legal_nearby_pos.append( 19*_y + _x )
            
        return legal_nearby_pos

    def GetBlock(self, pos, chi, visited):
        visited.add(pos)
        if self.board[pos] == SPACE:
            return set(),set()
        legal_nearby_pos = self.GetLegalNearby(pos)
        connected = []
        for newpos in legal_nearby_pos:
            if newpos in visited:
                continue
            if self.board[ newpos ] == self.board[pos]:
                connected.append(newpos)
            elif self.board[newpos] == SPACE:
                chi.add(newpos)

        for _pos in connected:
            self.GetBlock(_pos, chi, visited)
            
        return chi, visited
    
    def PrintBoard(self):

        for pos in range(len(self.board)):
            x = pos%19
            y = int(pos/19)
            c = ''

            if self.board[pos] == 1:
                c = 'X'
            elif self.board[pos] == 2:
                c = 'O'
            else:
                c = '.'
            print( c , end=' ')

            if x == 18:
                print()
        print()

    def GetBoard(self):
        black, white = [], []
        for pos in range(len(self.board)):
            x = pos % 19
            y = int(pos/19)

            if self.board[pos] == BLACK:
                black.append(1)
                white.append(0)
            elif self.board[pos] == WHITE:
                black.append(0)
                white.append(1)
            else:
                black.append(0)
                white.append(0)

        return np.array(black).reshape((19, 19)), np.array(white).reshape((19, 19))
    
    def __eq__(self, other):
        return other.board == self.board
    
    def __add__(self, move):
        newb = BOARD()
        newb.board = self.board.copy()
        newb.PutStone(move)
        return newb

class GAME:
    def __init__(self):
        self.FileName = '(NULL)'
        self.boards = []
        self.Moves = []

    def Parser(self, sgf_str):
        def make_tag_idx_pairs(tags, idx, tagname):
            
            if len(tags) != len(idx):
                print('ERROR')           
            
            return [ [tagname, idx[i], tags[i]] for i in range(len(idx))]
        
        self.sgf_formats = []
        
        self.sgf_str = sgf_str[:sgf_str.find(')')+1]
        TagBPattern = '[^AP]B(\[..\])+'
        TagWPattern = '[^AP]W(\[..\])+'
        TagCPattern = '[^AP]C\[([^\]]*)\]'
        self.BTag = [ x[1:3].lower() for x in re.findall(TagBPattern, self.sgf_str)]
        self.BIdx = [ m.start(0) for m in re.finditer(TagBPattern, self.sgf_str)]
        self.WTag = [ x[1:3].lower() for x in re.findall(TagWPattern, self.sgf_str)]
        self.WIdx = [ m.start(0) for m in re.finditer(TagWPattern, self.sgf_str)]        
        self.CTag = [ x for x in re.findall(TagCPattern, self.sgf_str) ]
        self.CIdx = [ m.start(0) for m in re.finditer(TagCPattern, self.sgf_str)]

        self.LeftparenthesisIdx = [ ('(', m.start(0) ,'(') for m in re.finditer('\(', self.sgf_str)]
        self.RightparenthesisIdx = [ (')', m.start(0),')') for m in re.finditer('\)', self.sgf_str)]

        self.chrono = []

        for Q in [ (self.BTag, self.BIdx, 'B'), (self.WTag, self.WIdx, 'W'), (self.CTag, self.CIdx, 'C')]:
            self.chrono.extend( make_tag_idx_pairs( Q[0], Q[1], Q[2]) ) # tagname , char_idx, tags_content
        
        for Q in [self.LeftparenthesisIdx, self.RightparenthesisIdx ]:
            self.chrono.extend( list(Q) )

        self.chrono = sorted(self.chrono, key=lambda x: x[1])

        self._chrono = []
        for i in range(len(self.chrono)-1):
            if self.chrono[i][0] in 'BW' :
                if i+1 == len(self.chrono):
                    self._chrono.append( [self.chrono[i], ['C','-1','None']] )
                elif self.chrono[i+1][0] in 'C':
                    self._chrono.append( [self.chrono[i], self.chrono[i+1]] )
                elif self.chrono[i+1][0] not in 'C':
                    self._chrono.append( [self.chrono[i], ['C','-1','None']] )
                else:
                    err(' may not appear')

        for mv_c in self._chrono :
            piece = mv_c[0]
            comment = mv_c[1]
            
            #if the move is pass, then drop it
            if 'http' in comment[2] or 'tt' in piece[2]:
                continue
            
            color = BLACK if piece[0] == 'B' else WHITE
            pos = charchar2pos(piece[2])
            self.Moves.append( MOVE( color, pos, comment = comment[2]) )

    def GetMoveNum(self):
        return len(self.Moves)

    def GetMove(self, MoveNum):
        return self.Moves[MoveNum]

    def PlayTo(self, MoveNum):
        board = BOARD()
        for i in range(0, MoveNum):
            board += self.GetMove(i)

        #board.PrintBoard()
        return board
    
def GetBoard(sgf_str):
    g = GAME()
    g.Parser(sgf_str)
    now = g.PlayTo(g.GetMoveNum())
    pre = g.PlayTo(g.GetMoveNum() - 1)
    Black1, White1 = now.GetBoard()
    Black2, White2 = pre.GetBoard()

    Board = np.array(now.board).reshape((19, 19))
    Position = g.Moves[g.GetMoveNum()-1].pos

    return Black1, White1, Black2, White2, Board, Position
