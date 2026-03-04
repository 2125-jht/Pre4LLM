# 动态规划（Dynamic Programming）

---

## 1. 爬楼梯（Climbing Stairs）

**题号**：070  
**思路**：dp[i] = dp[i-1] + dp[i-2]，斐波那契数列  
**代码**：
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
**复杂度**：时间 O(n)，空间 O(1)

---

## 2. 最大子数组和（Maximum Subarray）

**题号**：053  
**思路**：Kadane 算法，dp[i] = max(nums[i], dp[i-1] + nums[i])  
**代码**：
```python
def maxSubArray(nums):
    curr_sum = max_sum = nums[0]
    for num in nums[1:]:
        curr_sum = max(num, curr_sum + num)
        max_sum = max(max_sum, curr_sum)
    return max_sum
```
**复杂度**：时间 O(n)，空间 O(1)

---

## 3. 打家劫舍（House Robber）

**题号**：198  
**思路**：dp[i] = max(dp[i-1], dp[i-2] + nums[i])  
**代码**：
```python
def rob(nums):
    prev2, prev1 = 0, 0
    for num in nums:
        curr = max(prev1, prev2 + num)
        prev2, prev1 = prev1, curr
    return prev1
```
**复杂度**：时间 O(n)，空间 O(1)

---

## 4. 不同路径（Unique Paths）

**题号**：062  
**思路**：dp[i][j] = dp[i-1][j] + dp[i][j-1]  
**代码**：
```python
def uniquePaths(m, n):
    dp = [1] * n
    for _ in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    return dp[-1]
```
**复杂度**：时间 O(m·n)，空间 O(n)

---

## 5. 跳跃游戏（Jump Game）

**题号**：055  
**思路**：贪心，维护最远可达距离  
**代码**：
```python
def canJump(nums):
    max_reach = 0
    for i, jump in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + jump)
    return True
```
**复杂度**：时间 O(n)，空间 O(1)

---

## 6. 零钱兑换（Coin Change）

**题号**：322  
**思路**：完全背包，dp[i] = min(dp[i], dp[i-coin] + 1)  
**代码**：
```python
def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```
**复杂度**：时间 O(amount·n)，空间 O(amount)

---

## 7. 最长递增子序列（Longest Increasing Subsequence）

**题号**：300  
**思路**：dp[i] 表示以 nums[i] 结尾的最长递增子序列长度  
**代码**：
```python
def lengthOfLIS(nums):
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
```
**复杂度**：时间 O(n²)，空间 O(n)  
**优化**：二分查找可优化至 O(n·logn)

---

## 8. 最长公共子序列（Longest Common Subsequence）

**题号**：1143  
**思路**：dp[i][j] 表示 text1[0:i] 和 text2[0:j] 的最长公共子序列  
**代码**：
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
**复杂度**：时间 O(m·n)，空间 O(m·n)

---

## 9. 编辑距离（Edit Distance）

**题号**：072  
**思路**：dp[i][j] 表示 word1[0:i] 转换成 word2[0:j] 的最小操作数  
**代码**：
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
**复杂度**：时间 O(m·n)，空间 O(m·n)
