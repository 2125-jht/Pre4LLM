# 回溯（Backtracking）

---

## 1. 电话号码的字母组合（Letter Combinations of a Phone Number）

**题号**：017  
**难度**：中等

### 题目描述
给定一个仅包含数字 `2-9` 的字符串，返回所有它能表示的字母组合。答案可以按 **任意顺序** 返回。

给出数字到字母的映射如下（与电话按键相同）。注意 `1` 不对应任何字母。

```
2: abc    3: def
4: ghi    5: jkl    6: mno
7: pqrs   8: tuv    9: wxyz
```

### 示例
```
输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]

输入：digits = ""
输出：[]

输入：digits = "2"
输出：["a","b","c"]
```

### 解题思路
递归遍历每个数字对应的字母。

### 代码实现
```python
def letterCombinations(digits):
    if not digits:
        return []
    phone = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
             '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
    res = []
    
    def backtrack(index, path):
        if index == len(digits):
            res.append(''.join(path))
            return
        for char in phone[digits[index]]:
            path.append(char)
            backtrack(index + 1, path)
            path.pop()
    
    backtrack(0, [])
    return res
```

### 复杂度分析
- **时间复杂度**：O(3ⁿ·4ᵐ)，n 为 3 字母数字个数，m 为 4 字母数字个数
- **空间复杂度**：O(n)

---

## 2. 括号生成（Generate Parentheses）

**题号**：022  
**难度**：中等

### 题目描述
数字 `n` 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 **有效的** 括号组合。

### 示例
```
输入：n = 3
输出：["((()))","(()())","(())()","()(())","()()()"]

输入：n = 1
输出：["()"]
```

### 解题思路
回溯，左括号数 > 右括号数时可加右括号。

### 代码实现
```python
def generateParenthesis(n):
    res = []
    
    def backtrack(left, right, path):
        if left == right == n:
            res.append(''.join(path))
            return
        if left < n:
            path.append('(')
            backtrack(left + 1, right, path)
            path.pop()
        if right < left:
            path.append(')')
            backtrack(left, right + 1, path)
            path.pop()
    
    backtrack(0, 0, [])
    return res
```

### 复杂度分析
- **时间复杂度**：O(4ⁿ/√n)，卡特兰数
- **空间复杂度**：O(n)

---

## 3. 组合总和（Combination Sum）

**题号**：039  
**难度**：中等

### 题目描述
给你一个 **无重复元素** 的整数数组 `candidates` 和一个目标整数 `target`，找出 `candidates` 中可以使数字和为目标数 `target` 的所有 **不同组合**，并以列表形式返回。你可以按 **任意顺序** 返回这些组合。

`candidates` 中的 **同一个** 数字可以 **无限制重复被选取**。如果至少一个数字的被选数量不同，则两种组合是不同的。

### 示例
```
输入：candidates = [2,3,6,7], target = 7
输出：[[2,2,3],[7]]
解释：
2 和 3 可以形成一组候选，2 + 2 + 3 = 7 。注意 2 可以使用多次
7 也是一个候选， 7 = 7
仅有这两种组合

输入：candidates = [2,3,5], target = 8
输出：[[2,2,2,2],[2,3,3],[3,5]]
```

### 解题思路
回溯，允许重复使用元素，注意剪枝。

### 代码实现
```python
def combinationSum(candidates, target):
    res = []
    candidates.sort()
    
    def backtrack(start, target, path):
        if target == 0:
            res.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > target:
                break
            path.append(candidates[i])
            backtrack(i, target - candidates[i], path)  # i 不 +1，可重复
            path.pop()
    
    backtrack(0, target, [])
    return res
```

### 复杂度分析
- **时间复杂度**：O(n^(target/min))
- **空间复杂度**：O(target/min)

---

## 4. 全排列（Permutations）

**题号**：046  
**难度**：中等

### 题目描述
给定一个不含重复数字的数组 `nums`，返回其 **所有可能的全排列**。你可以 **按任意顺序** 返回答案。

### 示例
```
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

输入：nums = [0,1]
输出：[[0,1],[1,0]]
```

### 解题思路
回溯，用 visited 数组标记已使用元素。

### 代码实现
```python
def permute(nums):
    res = []
    n = len(nums)
    
    def backtrack(path, visited):
        if len(path) == n:
            res.append(path[:])
            return
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            path.append(nums[i])
            backtrack(path, visited)
            path.pop()
            visited[i] = False
    
    backtrack([], [False] * n)
    return res
```

### 复杂度分析
- **时间复杂度**：O(n·n!)
- **空间复杂度**：O(n)

---

## 5. 子集（Subsets）

**题号**：078  
**难度**：中等

### 题目描述
给你一个整数数组 `nums`，数组中的元素 **互不相同**。返回该数组所有可能的子集（幂集）。

解集 **不能** 包含重复的子集。你可以按 **任意顺序** 返回解集。

### 示例
```
输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

输入：nums = [0]
输出：[[],[0]]
```

### 解题思路
回溯，每个元素选或不选。

### 代码实现
```python
def subsets(nums):
    res = []
    
    def backtrack(start, path):
        res.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return res
```

### 复杂度分析
- **时间复杂度**：O(n·2ⁿ)
- **空间复杂度**：O(n)

---

## 6. 单词搜索（Word Search）

**题号**：079  
**难度**：中等

### 题目描述
给定一个 `m x n` 二维字符网格 `board` 和一个字符串单词 `word`。如果 `word` 存在于网格中，返回 `true`；否则，返回 `false`。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中"相邻"单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

### 示例
```
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true

输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
输出：true

输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
输出：false
```

### 解题思路
回溯 + DFS，标记访问过的格子。

### 代码实现
```python
def exist(board, word):
    m, n = len(board), len(board[0])
    
    def dfs(i, j, k):
        if k == len(word):
            return True
        if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != word[k]:
            return False
        
        temp = board[i][j]
        board[i][j] = '#'  # 标记已访问
        
        found = (dfs(i + 1, j, k + 1) or
                 dfs(i - 1, j, k + 1) or
                 dfs(i, j + 1, k + 1) or
                 dfs(i, j - 1, k + 1))
        
        board[i][j] = temp  # 恢复
        return found
    
    for i in range(m):
        for j in range(n):
            if dfs(i, j, 0):
                return True
    return False
```

### 复杂度分析
- **时间复杂度**：O(m·n·3ᴸ)，L 为单词长度
- **空间复杂度**：O(L)

---

## 7. N 皇后（N-Queens）

**题号**：051  
**难度**：困难

### 题目描述
按照国际象棋的规则，皇后可以攻击与之处在同一行或同一列或同一斜线上的棋子。

**n 皇后问题** 研究的是如何将 `n` 个皇后放置在 `n×n` 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 `n`，返回所有不同的 **n 皇后问题** 的解决方案。

每一种解法包含一个不同的 **n 皇后问题** 的棋子放置方案，该方案中 `'Q'` 和 `'.'` 分别代表了皇后和空位。

### 示例
```
输入：n = 4
输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
解释：4 皇后问题存在两个不同的解法

输入：n = 1
输出：[["Q"]]
```

### 解题思路
回溯，按行放置，检查列和对角线。

### 代码实现
```python
def solveNQueens(n):
    res = []
    board = [['.'] * n for _ in range(n)]
    
    def is_safe(row, col):
        for i in range(row):
            if board[i][col] == 'Q':
                return False
            if col - (row - i) >= 0 and board[i][col - (row - i)] == 'Q':
                return False
            if col + (row - i) < n and board[i][col + (row - i)] == 'Q':
                return False
        return True
    
    def backtrack(row):
        if row == n:
            res.append([''.join(r) for r in board])
            return
        for col in range(n):
            if is_safe(row, col):
                board[row][col] = 'Q'
                backtrack(row + 1)
                board[row][col] = '.'
    
    backtrack(0)
    return res
```

### 复杂度分析
- **时间复杂度**：O(n!)
- **空间复杂度**：O(n²)
