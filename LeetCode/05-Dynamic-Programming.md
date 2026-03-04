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

---

## 10. 最长回文子串（Longest Palindromic Substring）

**题号**：005  
**难度**：中等

### 题目描述
给你一个字符串 `s`，找到 `s` 中最长的 **回文子串**。

### 示例
```
输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案

输入：s = "cbbd"
输出："bb"
```

### 解题思路
中心扩展法，从每个位置向两边扩展；或动态规划。

### 代码实现
```python
def longestPalindrome(s):
    if not s:
        return ""
    
    start, end = 0, 0
    
    def expandAroundCenter(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1
    
    for i in range(len(s)):
        len1 = expandAroundCenter(i, i)       # 奇数长度
        len2 = expandAroundCenter(i, i + 1)   # 偶数长度
        max_len = max(len1, len2)
        if max_len > end - start:
            start = i - (max_len - 1) // 2
            end = i + max_len // 2
    
    return s[start:end+1]
```

### 复杂度分析
- **时间复杂度**：O(n²)
- **空间复杂度**：O(1)

---

## 11. 分割等和子集（Partition Equal Subset Sum）

**题号**：416  
**难度**：中等

### 题目描述
给你一个 **只包含正整数** 的 **非空** 数组 `nums`。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

### 示例
```
输入：nums = [1,5,11,5]
输出：true
解释：数组可以分割成 [1, 5, 5] 和 [11]

输入：nums = [1,2,3,5]
输出：false
解释：数组不能分割成两个元素和相等的子集
```

### 解题思路
0-1 背包问题，判断是否存在子集和为总和的一半。

### 代码实现
```python
def canPartition(nums):
    total = sum(nums)
    if total % 2 != 0:
        return False
    
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        for i in range(target, num - 1, -1):
            dp[i] = dp[i] or dp[i - num]
    
    return dp[target]
```

### 复杂度分析
- **时间复杂度**：O(n·target)
- **空间复杂度**：O(target)

---

## 12. 单词拆分（Word Break）

**题号**：139  
**难度**：中等

### 题目描述
给你一个字符串 `s` 和一个字符串列表 `wordDict` 作为字典。请你判断是否可以利用字典中出现的单词拼接出 `s`。

**注意**：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。

### 示例
```
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以由 "leet" 和 "code" 拼接成

输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
```

### 解题思路
`dp[i]` 表示 `s[0:i]` 是否可以被拆分。

### 代码实现
```python
def wordBreak(s, wordDict):
    word_set = set(wordDict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[n]
```

### 复杂度分析
- **时间复杂度**：O(n²)
- **空间复杂度**：O(n)

---

## 13. 乘积最大子数组（Maximum Product Subarray）

**题号**：152  
**难度**：中等

### 题目描述
给你一个整数数组 `nums`，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

测试用例的答案是一个 **32-位** 整数。

### 示例
```
输入: nums = [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6

输入: nums = [-2,0,-1]
输出: 0
解释: 结果不能为 2，因为 [-2,-1] 不是子数组
```

### 解题思路
同时维护最大和最小值（负数可能变成最大），遍历更新。

### 代码实现
```python
def maxProduct(nums):
    if not nums:
        return 0
    
    max_prod = min_prod = result = nums[0]
    
    for i in range(1, len(nums)):
        if nums[i] < 0:
            max_prod, min_prod = min_prod, max_prod
        
        max_prod = max(nums[i], max_prod * nums[i])
        min_prod = min(nums[i], min_prod * nums[i])
        result = max(result, max_prod)
    
    return result
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

---

## 14. 最长有效括号（Longest Valid Parentheses）

**题号**：032  
**难度**：困难

### 题目描述
给你一个只包含 `'('` 和 `')'` 的字符串，找出最长有效（匹配正确）括号子串的长度。

### 示例
```
输入：s = "(()"
输出：2
解释：最长有效括号子串是 "()"

输入：s = ")()())"
输出：4
解释：最长有效括号子串是 "()()"

输入：s = ""
输出：0
```

### 解题思路
栈存储索引，栈底元素为当前有效括号子串的前一个位置。

### 代码实现
```python
def longestValidParentheses(s):
    stack = [-1]  # 初始值，用于计算长度
    max_len = 0
    
    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)  # 新的基准位置
            else:
                max_len = max(max_len, i - stack[-1])
    
    return max_len
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(n)

---

## 15. 最佳买卖股票时机含冷冻期（Best Time to Buy and Sell Stock with Cooldown）

**题号**：309  
**难度**：中等

### 题目描述
给定一个整数数组 `prices`，其中第 `prices[i]` 表示某支股票第 `i` 天的价格。

在每一天，你可以决定是否购买和/或出售股票。你在任何时候 **最多** 只能持有 **一股** 股票。你也可以先购买，然后在 **同一天** 出售。

返回你能获得的 **最大** 利润。注意：卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。

### 示例
```
输入: prices = [1,2,3,0,2]
输出: 3
解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]
```

### 解题思路
三个状态：持有股票、不持有股票且处于冷冻期、不持有股票且不处于冷冻期。

### 代码实现
```python
def maxProfit(prices):
    if not prices:
        return 0
    
    n = len(prices)
    hold = [0] * n      # 持有股票
    sold = [0] * n      # 不持有，刚卖出（冷冻期）
    rest = [0] * n      # 不持有，可买
    
    hold[0] = -prices[0]
    
    for i in range(1, n):
        hold[i] = max(hold[i-1], rest[i-1] - prices[i])
        sold[i] = hold[i-1] + prices[i]
        rest[i] = max(rest[i-1], sold[i-1])
    
    return max(sold[-1], rest[-1])
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(n) - 可优化至 O(1)
