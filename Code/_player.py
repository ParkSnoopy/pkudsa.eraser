import numpy as np
from time import perf_counter

# 为了以 x y 形式使用坐标，而不以 [0] [1] 形式使用，这里定义 vec2
from collections import namedtuple
vec2 = namedtuple("vec2", ("x", "y"))

# Operation 是包含两个 vec2 对象的 tuple
# 表示 将 pos1 的物体移动到 pos2 的操作
Operation = namedtuple("Operation", ("pos1", "pos2"))

class Pattern:
    """
    包含
    pattern  ：可以进行操作的样板
    operation：上面 pattern 对应的操作（什么操作可以将上面的 pattern 消除）
    """
    def __init__(self, pattern, operation):
        self.pattern = pattern
        self.operation = operation

# 消除的标记为 NA，现在是 0
NA = 0

# 原来定义了探索的深度，后来用全部时间，尽量深地探索，deprecated
LEVEL = 1200//6

# 要演示的 Board 的大小
BOARDSIZE = vec2(6, 6*LEVEL)

# 为了下面 Pattern 定义方便，array=np.array
array = np.array

# 所有可以操作的 Pattern
#
# key  ：Pattern 的大小
# value：Pattern 的集合
PATTERNS: dict[vec2: set] = {
    vec2(3, 2): {
        # RIGHT
        Pattern(
            array((
                (1, 0), 
                (0, 1), 
                (0, 1), 
            )), 
            Operation(
                vec2(0, 0), 
                vec2(0, 1)
            )
        ), 
        Pattern(
            array((
                (0, 1), 
                (1, 0), 
                (0, 1), 
            )), 
            Operation(
                vec2(1, 0), 
                vec2(1, 1)
            )
        ), 
        Pattern(
            array((
                (0, 1), 
                (0, 1), 
                (1, 0), 
            )), 
            Operation(
                vec2(2, 0), 
                vec2(2, 1)
            )
        ), 
        # LEFT
        Pattern(
            array((
                (0, 1), 
                (1, 0), 
                (1, 0), 
            )), 
            Operation(
                vec2(0, 1), 
                vec2(0, 0)
            )
        ), 
        Pattern(
            array((
                (1, 0), 
                (0, 1), 
                (1, 0), 
            )), 
            Operation(
                vec2(1, 1), 
                vec2(1, 0)
            )
        ), 
        Pattern(
            array((
                (1, 0), 
                (1, 0), 
                (0, 1), 
            )), 
            Operation(
                vec2(2, 1), 
                vec2(2, 0)
            )
        ), 
    }, 
    vec2(1, 4): {
        # RIGHT
        Pattern(
            array((
                (1, 0, 1, 1), 
            )), 
            Operation(
                vec2(0, 0), 
                vec2(0, 1)
            )
        ), 
        # LEFT
        Pattern(
            array((
                (1, 1, 0, 1), 
            )), 
            Operation(
                vec2(0, 3), 
                vec2(0, 2)
            )
        ), 
    }, 
    vec2(2, 3): {
        # TOP
        Pattern(
            array((
                (0, 1, 1), 
                (1, 0, 0), 
            )), 
            Operation(
                vec2(1, 0), 
                vec2(0, 0)
            )
        ), 
        Pattern(
            array((
                (1, 0, 1), 
                (0, 1, 0), 
            )), 
            Operation(
                vec2(1, 1), 
                vec2(0, 1)
            )
        ), 
        Pattern(
            array((
                (1, 1, 0), 
                (0, 0, 1), 
            )), 
            Operation(
                vec2(1, 2), 
                vec2(0, 2)
            )
        ), 
        # BOTTOM
        Pattern(
            array((
                (1, 0, 0), 
                (0, 1, 1), 
            )), 
            Operation(
                vec2(0, 0), 
                vec2(1, 0)
            )
        ), 
        Pattern(
            array((
                (0, 1, 0), 
                (1, 0, 1), 
            )), 
            Operation(
                vec2(0, 1), 
                vec2(1, 1)
            )
        ), 
        Pattern(
            array((
                (0, 0, 1), 
                (1, 1, 0), 
            )), 
            Operation(
                vec2(0, 2), 
                vec2(1, 2)
            )
        ), 
    }, 
    vec2(4, 1): {
        # TOP
        Pattern(
            array((
                tuple((1, )), 
                tuple((1, )), 
                tuple((0, )), 
                tuple((1, )), 
            )), 
            Operation(
                vec2(3, 0), 
                vec2(2, 0)
            )
        ), 
        # BOTTOM
        Pattern(
            array((
                tuple((1, )), 
                tuple((0, )), 
                tuple((1, )), 
                tuple((1, )), 
            )), 
            Operation(
                vec2(0, 0), 
                vec2(1, 0)
            )
        ), 
    }, 
}
def direction(boardshape: vec2, pos: vec2):
    """
    探索所有可以连消的方向的函数

    Parameters
    ----------
    boardshape : vec2
        要消除的 Board 大小（ 一般是 vec2(6,6) ）.
    pos : vec2
        当前的位置.

    Returns
    -------
    base : set[vec2]
        所有可以移动的方向（如果 pos 是 vec(0,0)，boardshape 是 vec(6,6) 的话，返回 { vec2(1,0), vec2(0,1) }.

    """
    base = {
        vec2( 1,  0), 
        vec2(-1,  0), 
        vec2( 0,  1), 
        vec2( 0, -1), 
    }
    if ( pos.x + 1 ) == boardshape.x:
        base.remove(
            vec2(1, 0)
        )
    if ( pos.y + 1 ) == boardshape.y:
        base.remove(
            vec2(0, 1)
        )
    if pos.x == 0:
        base.remove(
            vec2(-1, 0)
        )
    if pos.y == 0:
        base.remove(
            vec2(0, -1)
        )
    return base

def direction_conn(boardshape: vec2, pos: vec2, connectivity: int):
    """
    （标记用的函数，之查找 右、下 方向）为了发现相邻可以移动的样板，查找所有 x y 轴相邻 以及 对角方向相邻

    Parameters
    ----------
    boardshape : vec2
    pos : vec2
        （与 direction 中的 boardshape 和 pos 相同）.
    connectivity : int
        连接程度（=1：直角方向的连接；=2：包括对角方向的连接）.

    Returns
    -------
    base : set[vec2]
        （与 direction 相同）.

    """
    base = {
        vec2(-1,  0), 
        vec2( 0, -1), 
    }
    if connectivity == 2:
        base.update({
            vec2(-1,  1), 
            vec2(-1, -1), 
        })
        if pos.x == 0:
            base.remove(
                vec2(-1, 0)
            )
            base.remove(
                vec2(-1, 1)
            )
            base.remove(
                vec2(-1, -1)
            )
        if ( pos.y + 1 ) == boardshape.y:
            if pos.x != 0:
                base.remove(
                    vec2(-1, 1)
                )
        if pos.y == 0:
            base.remove(
                vec2(0, -1)
            )
            if pos.x != 0:
                base.remove(
                    vec2(-1, -1)
                )
        return base
    if pos.x == 0:
        base.remove(
            vec2(-1, 0)
        )
    if pos.y == 0:
        base.remove(
            vec2(0, -1)
        )
    return base

class ScoredBoard:
    """
    保存 Board 与其 状态
    """
    def __init__(self, select_hist, score_hist, score, board, operation):
        """
        Parameters
        ----------
        select_hist : list
            到达这个 Board 的选择历史.
        score_hist : list
            到达这个 Board 的分数历史.
        score : int
            当前自己的分数.
        board : np.ndarray
            游戏盘.
        operation : Operation
            从上一个 Board 到这个 Board 需要进行的操作.
        """
        self.select_hist = select_hist
        self.score_hist = score_hist
        self.score = score
        self.board = board
        self.operation = operation
    def __repr__(self):
        return f"ScoreBoard(\nselect_hist={self.select_hist}, \nscore_hist={self.score_hist}, \nscore={self.score}, \nboard=\n{self.board}, \noperation=\n{self.operation}\n)"

class Plaser:
    """
    角色
    
    主要缺陷：不能模拟连消（一次消除以后，不能识别 还可以消除的状态）
              （时间紧迫，暂时放弃实现了）
    """
    
    def __init__(self, is_first: bool = None) -> None:
        self.is_first = is_first
    
    # scoreboards：每个深度 所有可能性的 list
    scoreboards: list[set[ScoredBoard]] = list()
    # 转换 Board 中的元素（现在不用转换，但其他代码已经假设 int 写了，怕出现错误）
    mapper = {
        'R': 1, 
        'B': 2, 
        'G': 3, 
        'Y': 4, 
        'P': 5, 
    }
    
    def _init(self, board, availables):
        """
        每次自己的顺序 进行的 init
        """
        
        observation: np.ndarray = board[:BOARDSIZE.x, :BOARDSIZE.y]
        
        for key, value in self.mapper.items():
            observation[ observation == key ] = value
        # print(f"\n  observation=\n{observation}\n")
        observation[ observation == str(np.nan) ] = '0'
        # print(f"\n  observation=\n{observation}\n")

        observation = observation.astype(np.int8)
        
        self.scoreboards = [[
            ScoredBoard(
                [], 
                [], 
                0, 
                observation, 
                None, 
            ), 
        ]]
        
    def _find_next_level(self, select: int, scoreboard: ScoredBoard) -> list[np.ndarray]:
        """
        查找下一个深度的所有可能性

        Parameters
        ----------
        select : int
            当前所有可能性中选择的 Board 序号.
        scoreboard : ScoredBoard
            要找 下一个深度 可能性的 Board 的 ScoredBoard.

        Returns
        -------
        list[ScoredBoard]
            当前选择的 Board 的所有下一个深度的可能性.

        """
        fullboard = scoreboard.board
        board = fullboard[:BOARDSIZE.x, :BOARDSIZE.x]
        
        possibles = self._find_possible(board)
        
        return [
            self._move(select, scoreboard, board, operation) for operation in possibles
        ]
    
    def _move(self, select: int, scoreboard: ScoredBoard, board: np.ndarray, operation: Operation) -> ScoredBoard:
        """
        返回 在 board 上进行 operation 操作的结果

        Parameters
        ----------
        select : int
            为了在新生成的 ScoredBoard 中记录当前历史.
        scoreboard : ScoredBoard
            为了在新生成的 ScoredBoard 中记录以前历史.
        board : np.ndarray
            目标的 Board.
        operation : Operation
            进行的操作.

        Returns
        -------
        ScoredBoard
            在 board 上进行 operation 操作的结果 的 ScoredBoard.

        """
        pos1, pos2 = operation.pos1, operation.pos2
        
        new = board.copy()
        
        new[pos1] = board[pos2]
        new[pos2] = board[pos1]
        
        erasedboard, erased = self._erase(new, pos2)
        
        fullboard = self._update_fullboard(scoreboard.board, erasedboard)
        
        erased_score = self._heuristic(erased)
        
        return ScoredBoard(
            scoreboard.select_hist + [select], 
            scoreboard.score_hist + [erased_score], 
            scoreboard.score + ( 0 if self.level % 2 else erased_score ), 
            fullboard, 
            operation, 
        )
    
    def _update_fullboard(self, fullboard: np.ndarray, board: np.ndarray):
        """
        将 6x1200 的 Board 信息更新

        Parameters
        ----------
        fullboard : np.ndarray
            6x1200 的棋盘.
        board : np.ndarray
            消除操作以后的 6x6 棋盘.

        Returns
        -------
        fullboard : TYPE
            将所有消除元素以后的空间填满以后的 6x1200 棋盘.

        """
        fullboard = fullboard.copy()
        fullboard[:BOARDSIZE.x, :BOARDSIZE.x] = board
        
        for i in range(BOARDSIZE.x):
            
            fullboard[i] = np.concatenate((
                fullboard[i][ fullboard[i]!=NA ], 
                np.zeros(
                    (BOARDSIZE.y - (fullboard[i][ fullboard[i]!=NA ]).shape[0], ), 
                    dtype=np.int8
                )
            ))
        
        return fullboard
    
    def _erase(self, board: np.ndarray, pos: vec2, erased=0) -> [np.ndarray, int]:
        """
        递归消除相邻的相同元素

        Parameters
        ----------
        board : np.ndarray
            6x6 的主棋盘.
        pos : vec2
            消除开始的位置（一般是 Operation.pos2）.
        erased : int, optional
            使用者不用输入，递归时使用，指当前以消除的个数.

        Returns
        -------
        board : np.ndarray
            元素消除以后标记为 NA 的 6x6 的主棋盘.
        erased : int
            消除的元素个数.

        """
        target = board[pos]
        board[pos] = NA
        erased += 1
        for d in direction(vec2(*board.shape), pos):
            newpos = vec2(
                pos.x + d.x, 
                pos.y + d.y
            )
            if board[newpos] == target:
                board, erased = self._erase(board, newpos, erased=erased)
        return board, erased
        
    def _find_possible(self, board: np.ndarray) -> list[Operation]:
        """
        查找当前可以操作的所有 Operation

        Parameters
        ----------
        board : np.ndarray
            6x6 主棋盘.

        Returns
        -------
        list[Operation]
            当前主棋盘能进行的所有操作.

        """
        labeled = self._label(board, connectivity=2)
        
        clustered_indexes = [
            np.where(labeled == label)
            for label in np.unique(labeled)
            if label != NA
        ]
        
        return self._filter_possible(clustered_indexes)
    
    def _label(self, board: np.ndarray, connectivity: int) -> np.ndarray:
        """
        标记 6x6 主棋盘的相邻元素

        Parameters
        ----------
        board : np.ndarray
            6x6 主棋盘
        connectivity : int
            连接度.

        Returns
        -------
        new : np.ndarray
            标记后的 6x6 主棋盘.

        """
        no = 1
        boardshape = vec2(*board.shape)
        new = np.zeros(boardshape, dtype=np.int8)
        for element in self.mapper.values():
            is_element: np.ndarray = np.array( board == element , dtype=np.int8 )
            _x, _y = is_element.shape
            for x in range(_x):
                for y in range(_y):
                    labeled = False
                    if is_element[x][y]:
                        for d in direction_conn(boardshape, vec2(x,y), connectivity):
                            if is_element[x+d.x][y+d.y]:
                                is_element[x][y] = is_element[x+d.x][y+d.y]
                                labeled = True
                                # print(is_element)
                                break
                        if not labeled:
                            is_element[x][y] = no
                            no += 1
            # print(f"\nnew=\n{new}\n")
            new = new + is_element
        return new
    
    def _filter_possible(self, clustered_indexes: list[np.ndarray, np.ndarray]) -> list[Operation]:
        """
        （有点复杂啰嗦的函数，没时间简化了）
        输入 所有标记（ _label() ）的信息，过滤可以进行操作的部分，输出对应操作

        Parameters
        ----------
        clustered_indexes : list[np.ndarray, np.ndarray]
            所有标记（ _label() 返回信息的基本加工 ）的信息.

        Returns
        -------
        list[Operation]
            根据 PATTERNS 过滤出的所有可以进行操作的部分，对应的 Operation （ 需要把哪里的元素移到哪里 的操作 ）.

        """
        possibles = []
        for i, clustered_index in enumerate(clustered_indexes):
            if self.halt:
                break
            
            if len( clustered_index[0] ) >= 3:
                
                xmin, xmax, ymin, ymax = (
                    np.min(clustered_index[0]), 
                    np.max(clustered_index[0]), 
                    np.min(clustered_index[1]), 
                    np.max(clustered_index[1]), 
                )
                
                cluster_pattern = np.zeros((xmax-xmin+1, ymax-ymin+1), dtype=np.int8)
                
                for x, y in zip(*clustered_index):
                    cluster_pattern[x-xmin, y-ymin] = 1
                
                cluster_x, cluster_y = cluster_pattern.shape
                
                for shape, patterns in PATTERNS.items():
                    
                    if ( ( perf_counter() - self.t0 ) > 0.09 ):
                        self.halt = True
                        break
                    
                    if shape.x <= cluster_x and shape.y <= cluster_y:
                        
                        for pattern in patterns:
                            
                            flatpattern = pattern.pattern.flatten()
                            
                            for x in range( cluster_x - shape.x + 1 ):
                                for y in range( cluster_y - shape.y + 1 ):
                                    
                                    current = cluster_pattern[x:x+shape.x, y:y+shape.y]
                                    
                                    if ( flatpattern == current.flatten() ).all():
                                        operation = Operation(
                                            vec2(
                                                pattern.operation.pos1.x + x + xmin, 
                                                pattern.operation.pos1.y + y + ymin
                                            ), 
                                            vec2(
                                                pattern.operation.pos2.x + x + xmin, 
                                                pattern.operation.pos2.y + y + ymin
                                            ), 
                                        )
                                        possibles.append(
                                            ( operation, pattern.pattern )
                                        )
        
        return possibles
    
    def _heuristic(self, erased: int) -> int:
        """
        将 消除的元素个数转换成分数

        Parameters
        ----------
        erased : int
            消除的元素个数.

        Returns
        -------
        int
            对应的分数.

        """
        return ( erased - 2 ) ** 2
    
    def _traceback(self) -> Operation:
        """
        查找最好的操作，返回第一次进行的操作

        Returns
        -------
        Operation
            为了达到当前棋盘，第一次需要进行的操作.

        """
        # print(f"\n{self.scoreboards = }\n")
        
        best: ScoredBoard = None
        for i in range(len(self.scoreboards)-1, -1, -1):
            best: ScoredBoard = self.scoreboards[i][0]
            if best.operation:
                break
        
        if best.select_hist:
            root: ScoredBoard = self.scoreboards[1][best.select_hist[0]]
        else:
            root = self.scoreboards[0][0]
        
        return root.operation
    
    @staticmethod
    def _convert(operation: Operation, availables: list[ [tuple, tuple] ]) -> [tuple, tuple]:
        """
        将 Operation 转换成 pkudsa.eraser 能理解的形式

        Parameters
        ----------
        operation : Operation
            要转换的操作.
        availables : list[ [tuple, tuple] ]
            get_info() 返回的所有可能性.

        Returns
        -------
        [tuple, tuple]
            operation 转换成的 availables 中的一个.

        """
        operation = (
            (
                operation.pos1.x, 
                operation.pos1.y
            ), 
            (
                operation.pos2.x, 
                operation.pos2.y
            )
        )
        if operation in availables:
            return operation
        else:
            return operation[::-1]
    
    @staticmethod
    def _pack_operation(t: [tuple, tuple]) -> Operation:
        """
        将 get_info() 给出的 ((),()) 转换成 Operation 形式

        Parameters
        ----------
        t : [tuple, tuple]
            ((),()).

        Returns
        -------
        Operation
            对应的操作.

        """
        return Operation(
            vec2(*t[0]), 
            vec2(*t[1])
        )
    
    def _perform_onestep_best(self, board: np.ndarray, availables: list[tuple[tuple, tuple]]) -> [tuple, tuple]:
        """
        （因为当前我的 Plaser 不完善，有时出现 bug）
        出现异常时，进行 深度=1 的情况最好的结果

        Parameters
        ----------
        board : np.ndarray
            6x6 主棋盘.
        availables : list[tuple[tuple, tuple]]
            get_info() 给出的 所有可能性.

        Returns
        -------
        [tuple, tuple]
            一层深度 最好的结果.

        """
        onesteps = []
        targetboard = board[:BOARDSIZE.x, :BOARDSIZE.x]
        for available in availables:
            operation = self._pack_operation(available)
            
            board1, erased1 = self._erase(targetboard.copy(), operation.pos1)
            onesteps.append(
                ScoredBoard([], [], self._heuristic(erased1), board1, operation)
            )
            
            board2, erased2 = self._erase(targetboard.copy(), operation.pos2)
            onesteps.append(
                ScoredBoard([], [], self._heuristic(erased2), board2, operation)
            )
        
        onesteps.sort(key=lambda x: x.score, reverse=True)
        # print(f"\n{onesteps[:2] = }\n")
        # print(f"\n{availables = }\n")
        best = self._convert( onesteps[0].operation, availables )
        
        return best
    
    random = __import__("random")
    def _random(self, availables: list[ [tuple, tuple] ]) -> [tuple, tuple]:
        """
        最后的最后，异常发生时，进行 random 的操作了

        Parameters
        ----------
        availables : list[ [tuple, tuple] ]
            get_info() 给出的 所有可能性.

        Returns
        -------
        [tuple, tuple]
            随机的一个.

        """
        return self.random.choice(availables)
    
    def _run(self):
        """
        到时间限制，模拟尽量深的结果
        """
        self.halt = False
        self.level = -1
        while ( not self.halt ):
            if self.scoreboards[ self.level + 1 ] == []:
                break
            self.level += 1
            
            self.scoreboards.append( list() )
            
            for select, scoreboard in enumerate( self.scoreboards[self.level] ):
                self.scoreboards[ self.level + 1 ].extend(
                    self._find_next_level(select, scoreboard)
                )
            
            self.scoreboards[ self.level + 1 ] = sorted(
                self.scoreboards[ self.level + 1 ], 
                key=lambda scoreboard: sum(scoreboard.score_hist), 
                reverse=True
            )[:( -1 if self.level % 2 else 1)]
            
            if ( ( perf_counter() - self.t0 ) > .09 ):
                break
        
        if not self.scoreboards[ self.level + 1 ]:
            self.scoreboards.pop(self.level + 1)
    
    def move(self, board, availables, score, turn):
        """
        Game 叫出的函数

        Parameters
        ----------
        board : TYPE
            get_info()[0].
        availables : TYPE
            get_info()[1].
        score : TYPE
            （掉弃）.
        turn : TYPE
            （掉弃）.

        Returns
        -------
        [tuple, tuple]
            availables 中的一个.

        """
        try:
            self.t0 = perf_counter()
            
            self._init(board, availables)
            self._run()
            
            operation = self._traceback()
            operation = self._convert(operation, availables)
            
            return operation
            
        except AttributeError:
            onestep_best = self._perform_onestep_best(self.scoreboards[0][0].board, availables)
            if onestep_best:
                return onestep_best
            
            else:
                return self._random(availables)
        except:
            return self._random(availables)
