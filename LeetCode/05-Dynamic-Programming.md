# 动态规划（Dynamic Programming）

---

## 1. 爬楼梯（Climbing Stairs）

**题号**：070  
**难度**：简单

### 题目描述
假设你正在爬楼梯。需要 `n` 阶你才能到达楼顶。

每次你可以爬 `1` 或 `2` 个台阶。你有多少种不同的方法可以爬到楼顶呢？

### 示例
```
输入：n = 2
输出：2
解释：有两种方法可以爬到楼顶
1. 1 阶 + 1 阶
2. 2 阶

输入：n = 3
输出：3
解释：有三种方法可以爬到楼顶
1. 1 阶 + 1 阶 + 1 阶
2. 1 阶 + 2 阶
3. 2 阶 + 1 阶
```

### 解题思路
`dp[i] = dp[i-1] + dp[i-2]`，斐波那契数列。

### 代码实现
```python
def climbStairs(n):
    if n <= 2:
        return n
    prev2, prev1 = 1, 2
    for _ in range(3, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr
    return prev1
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

---

## 2. 最大子数组和（Maximum Subarray）

**题号**：053  
**难度**：中等

### 题目描述
给你一个整数数组 `nums`，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**子数组** 是数组中的一个连续部分。

### 示例
```
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6

输入：nums = [1]
输出：1
```

### 解题思路
Kadane 算法，`dp[i] = max(nums[i], dp[i-1] + nums[i])`。

### 代码实现
```python
def maxSubArray(nums):
    curr_sum = max_sum = nums[0]
    for num in nums[1:]:
        curr_sum = max(num, curr_sum + num)
        max_sum = max(max_sum, curr_sum)
    return max_sum
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

---

## 3. 打家劫舍（House Robber）

**题号**：198  
**难度**：中等

### 题目描述
你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警**。

给定一个代表每个房屋存放金额的非负整数数组，计算你 **不触动警报装置的情况下** ，今晚能够偷窃到的最高金额。

### 示例
```
输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)
偷窃到的最高金额 = 1 + 3 = 4

输入：[2,7,9,3,1]
输出：12
解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)
偷窃到的最高金额 = 2 + 9 + 1 = 12
```

### 解题思路
`dp[i] = max(dp[i-1], dp[i-2] + nums[i])`，选或不选当前房屋。

### 代码实现
```python
def rob(nums):
    prev2, prev1 = 0, 0
    for num in nums:
        curr = max(prev1, prev2 + num)
        prev2, prev1 = prev1, curr
    return prev1
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

---

## 4. 不同路径（Unique Paths）

**题号**：062  
**难度**：中等

### 题目描述
一个机器人位于一个 `m x n` 网格的左上角（起始点在下图中标记为 "Start"）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 "Finish"）。

问总共有多少条不同的路径？

### 示例
```
输入：m = 3, n = 7
输出：28

输入：m = 3, n = 2
输出：3
解释：
从左上角开始，总共有 3 条路径可以到达右下角：
1. 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右
3. 向下 -> 向右 -> 向下
```

### 解题思路
`dp[i][j] = dp[i-1][j] + dp[i][j-1]`，只能从上或左过来。

### 代码实现
```python
def uniquePaths(m, n):
    dp = [1] * n
    for _ in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    return dp[-1]
```

### 复杂度分析
- **时间复杂度**：O(m·n)
- **空间复杂度**：O(n)

---

## 5. 跳跃游戏（Jump Game）

**题号**：055  
**难度**：中等

### 题目描述
给定一个非负整数数组 `nums`，你最初位于数组的 **第一个下标** 。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标。

### 示例
```
输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1，然后再从下标 1 跳 3 步到达最后一个下标

输入：nums = [3,2,1,0,4]
输出：false
解释：无论怎样，你总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0，所以你永远不可能到达最后一个下标
```

### 解题思路
贪心，维护最远可达距离。

### 代码实现
```python
def canJump(nums):
    max_reach = 0
    for i, jump in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + jump)
    return True
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

---

## 6. 零钱兑换（Coin Change）

**题号**：322  
**难度**：中等

### 题目描述
给你一个整数数组 `coins`，表示不同面额的硬币；以及一个整数 `amount`，表示总金额。

计算并返回可以凑成总金额所需的 **最少的硬币个数**。如果没有任何一种硬币组合能组成总金额，返回 `-1`。

你可以认为每种硬币的数量是无限的。

### 示例
```
输入：coins = [1,2,5], amount = 11
输出：3
解释：11 = 5 + 5 + 1

输入：coins = [2], amount = 3
输出：-1
```

### 解题思路
完全背包问题，`dp[i] = min(dp[i], dp[i-coin] + 1)`。

### 代码实现
```python
def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

### 复杂度分析
- **时间复杂度**：O(amount·n)
- **空间复杂度**：O(amount)

---

## 7. 最长递增子序列（Longest Increasing Subsequence）

**题号**：300  
**难度**：中等

### 题目描述
给你一个整数数组 `nums`，找到其中最长严格递增子序列的长度。

**子序列** 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。

### 示例
```
输入：nums = [10,9,2,5,3,7,101,18]
输出：4
解释：最长递增子序列是 [2,3,7,101]，因此长度为 4

输入：nums = [0,1,0,3,2,3]
输出：4
```

### 解题思路
`dp[i]` 表示以 `nums[i]` 结尾的最长递增子序列长度。

### 代码实现
```python
def lengthOfLIS(nums):
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
```

### 复杂度分析
- **时间复杂度**：O(n²)
- **空间复杂度**：O(n)

**优化**：二分查找可优化至 O(n·logn)。

---

## 8. 最长公共子序列（Longest Common Subsequence）

**题号**：1143  
**难度**：中等

### 题目描述
给定两个字符串 `text1` 和 `text2`，返回这两个字符串的最长 **公共子序列** 的长度。如果不存在公共子序列，返回 `0`。

一个字符串的 **子序列** 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

### 示例
```
输入：text1 = "abcde", text2 = "ace"
输出：3
解释：最长公共子序列是 "ace"，它的长度为 3

输入：text1 = "abc", text2 = "abc"
输出：3
```

### 解题思路
`dp[i][j]` 表示 `text1[0:i]` 和 `text2[0:j]` 的最长公共子序列。

### 代码实现
```python
def longestCommonSubsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
```

### 复杂度分析
- **时间复杂度**：O(m·n)
- **空间复杂度**：O(m·n)

---

## 9. 编辑距离（Edit Distance）

**题号**：072  
**难度**：困难

### 题目描述
给你两个单词 `word1` 和 `word2`，请返回将 `word1` 转换成 `word2` 所使用的最少操作数。

你可以对一个单词进行如下三种操作：
- 插入一个字符
- 删除一个字符
- 替换一个字符

### 示例
```
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')

输入：word1 = "intention", word2 = "execution"
输出：5
```

### 解题思路
`dp[i][j]` 表示 `word1[0:i]` 转换成 `word2[0:j]` 的最小操作数。

### 代码实现
```python
def minDistance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],    # 删除
                                   dp[i][j-1],    # 插入
                                   dp[i-1][j-1])  # 替换
    return dp[m][n]
```

### 复杂度分析
- **时间复杂度**：O(m·n)
- **空间复杂度**：O(m·n)
