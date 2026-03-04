# 回溯（Backtracking）

---

## 1. 电话号码的字母组合（Letter Combinations of a Phone Number）

**题号**：017  
**思路**：递归遍历每个数字对应的字母  
**代码**：
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
**复杂度**：时间 O(3ⁿ·4ᵐ)，空间 O(n)（n为3字母数字数，m为4字母数字数）

---

## 2. 括号生成（Generate Parentheses）

**题号**：022  
**思路**：回溯，左括号数 > 右括号数时可加右括号  
**代码**：
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
**复杂度**：时间 O(4ⁿ/√n)，空间 O(n)

---

## 3. 组合总和（Combination Sum）

**题号**：039  
**思路**：回溯，允许重复使用元素，注意剪枝  
**代码**：
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
**复杂度**：时间 O(n^(target/min))，空间 O(target/min)

---

## 4. 全排列（Permutations）

**题号**：046  
**思路**：回溯，用 visited 数组标记已使用元素  
**代码**：
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
**复杂度**：时间 O(n·n!)，空间 O(n)

---

## 5. 子集（Subsets）

**题号**：078  
**思路**：回溯，每个元素选或不选  
**代码**：
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
**复杂度**：时间 O(n·2ⁿ)，空间 O(n)

---

## 6. 单词搜索（Word Search）

**题号**：079  
**思路**：回溯 + DFS，标记访问过的格子  
**代码**：
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
**复杂度**：时间 O(m·n·3ᴸ)，空间 O(L)（L为单词长度）

---

## 7. N 皇后（N-Queens）

**题号**：051  
**思路**：回溯，按行放置，检查列和对角线  
**代码**：
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
**复杂度**：时间 O(n!)，空间 O(n²)
