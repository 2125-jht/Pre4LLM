# 图论（Graph）

---

## 1. 岛屿数量（Number of Islands）

**题号**：200  
**难度**：中等

### 题目描述
给你一个由 `'1'`（陆地）和 `'0'`（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

### 示例
```
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1

输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3
```

### 解题思路
DFS/BFS，遍历到 `'1'` 时沉岛计数。

### 代码实现（DFS）
```python
def numIslands(grid):
    if not grid:
        return 0
    m, n = len(grid), len(grid[0])
    count = 0
    
    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] == '0':
            return
        grid[i][j] = '0'  # 沉岛
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                dfs(i, j)
                count += 1
    return count
```

### 复杂度分析
- **时间复杂度**：O(m·n)
- **空间复杂度**：O(m·n)

---

## 2. 课程表（Course Schedule）

**题号**：207  
**难度**：中等

### 题目描述
你这个学期必须选修 `numCourses` 门课程，记为 `0` 到 `numCourses - 1` 。

在选修某些课程之前需要一些先修课程。先修课程按数组 `prerequisites` 给出，其中 `prerequisites[i] = [ai, bi]`，表示如果要学习课程 `ai` 则 **必须** 先学习课程 `bi`。

- 例如，先修课程对 `[0, 1]` 表示：想要学习课程 `0`，你需要先完成课程 `1`。

请你判断是否可能完成所有课程的学习？如果可以，返回 `true`；否则，返回 `false`。

### 示例
```
输入：numCourses = 2, prerequisites = [[1,0]]
输出：true
解释：总共有 2 门课程。学习课程 1 之前，你需要完成课程 0。这是可能的

输入：numCourses = 2, prerequisites = [[1,0],[0,1]]
输出：false
解释：总共有 2 门课程。学习课程 1 之前，你需要先完成课程 0；并且学习课程 0 之前，你还应先完成课程 1。这是不可能的
```

### 解题思路
拓扑排序，检测有向图是否有环。

### 代码实现（Kahn 算法）
```python
def canFinish(numCourses, prerequisites):
    from collections import defaultdict, deque
    
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    
    for a, b in prerequisites:
        graph[b].append(a)
        in_degree[a] += 1
    
    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    visited = 0
    
    while queue:
        course = queue.popleft()
        visited += 1
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)
    
    return visited == numCourses
```

### 复杂度分析
- **时间复杂度**：O(V + E)，V 为课程数，E 为先修关系数
- **空间复杂度**：O(V + E)

---

## 3. 实现 Trie（前缀树）（Implement Trie）

**题号**：208  
**难度**：中等

### 题目描述
**Trie**（发音类似 "try"）或者说 **前缀树** 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，如自动补全和拼写检查。

请你实现 Trie 类：
- `Trie()` 初始化前缀树对象
- `void insert(String word)` 向前缀树中插入字符串 `word`
- `boolean search(String word)` 如果字符串 `word` 在前缀树中，返回 `true`；否则，返回 `false`
- `boolean startsWith(String prefix)` 如果之前已经插入的字符串 `word` 的前缀之一为 `prefix`，返回 `true`；否则，返回 `false`

### 示例
```
输入：
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
输出：
[null, null, true, false, true, null, true]

解释：
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // 返回 True
trie.search("app");     // 返回 False
trie.startsWith("app"); // 返回 True
trie.insert("app");
trie.search("app");     // 返回 True
```

### 解题思路
多叉树，每个节点存储子节点和是否为单词结尾。

### 代码实现
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self._find_node(word)
        return node is not None and node.is_end
    
    def startsWith(self, prefix):
        return self._find_node(prefix) is not None
    
    def _find_node(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
```

### 复杂度分析
- **insert**：O(m)，m 为单词长度
- **search**：O(m)
- **startsWith**：O(m)
- **空间复杂度**：O(n·m)，n 为单词数

---

## 4. 腐烂的橘子（Rotting Oranges）

**题号**：994  
**难度**：中等

### 题目描述
在给定的 `m x n` 网格 `grid` 中，每个单元格可以有以下三个值之一：
- `0` 代表空单元格
- `1` 代表新鲜橘子
- `2` 代表腐烂的橘子

每分钟，腐烂的橘子 **周围 4 个方向上相邻** 的新鲜橘子都会腐烂。

返回直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 `-1`。

### 示例
```
输入：grid = [[2,1,1],[1,1,0],[0,1,1]]
输出：4

输入：grid = [[2,1,1],[0,1,1],[1,0,1]]
输出：-1
解释：左下角的橘子（第 2 行， 第 0 列）永远不会腐烂，因为腐烂只会从 4 个方向发生

输入：grid = [[0,2]]
输出：0
解释：因为 0 分钟时已经没有新鲜橘子了，所以答案就是 0
```

### 解题思路
BFS，多源扩散，统计分钟数。

### 代码实现
```python
def orangesRotting(grid):
    from collections import deque
    m, n = len(grid), len(grid[0])
    queue = deque()
    fresh = 0
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 2:
                queue.append((i, j))
            elif grid[i][j] == 1:
                fresh += 1
    
    minutes = 0
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while queue and fresh > 0:
        minutes += 1
        for _ in range(len(queue)):
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == 1:
                    grid[nx][ny] = 2
                    fresh -= 1
                    queue.append((nx, ny))
    
    return minutes if fresh == 0 else -1
```

### 复杂度分析
- **时间复杂度**：O(m·n)
- **空间复杂度**：O(m·n)
