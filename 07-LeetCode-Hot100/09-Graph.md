# 图论（Graph）

---

## 1. 岛屿数量（Number of Islands）

**题号**：200  
**思路**：DFS/BFS，遍历到 '1' 时沉岛计数  
**代码**（DFS）：
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
**复杂度**：时间 O(m·n)，空间 O(m·n)

---

## 2. 课程表（Course Schedule）

**题号**：207  
**思路**：拓扑排序，检测有向图是否有环  
**代码**（Kahn 算法）：
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
**复杂度**：时间 O(V + E)，空间 O(V + E)

---

## 3. 实现 Trie（前缀树）（Implement Trie）

**题号**：208  
**思路**：多叉树，每个节点存储子节点和是否为单词结尾  
**代码**：
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
**复杂度**：insert O(m)，search O(m)，startsWith O(m)，空间 O(n·m)

---

## 4. 腐烂的橘子（Rotting Oranges）

**题号**：994  
**思路**：BFS，多源扩散，统计分钟数  
**代码**：
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
**复杂度**：时间 O(m·n)，空间 O(m·n)
