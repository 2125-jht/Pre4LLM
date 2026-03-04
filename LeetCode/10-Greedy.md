# 贪心（Greedy）

---

## 1. 跳跃游戏 II（Jump Game II）

**题号**：045  
**难度**：中等

### 题目描述
给定一个长度为 `n` 的 **0 索引** 整数数组 `nums`。初始位置为 `nums[0]`。

每个元素 `nums[i]` 表示从索引 `i` 向前跳转的最大长度。换句话说，如果你在 `nums[i]` 处，你可以跳转到任意 `nums[i + j]` 处:
- `0 <= j <= nums[i]` 
- `i + j < n`

返回到达 `nums[n - 1]` 的最小跳跃次数。生成的测试用例可以到达 `nums[n - 1]`。

### 示例
```
输入: nums = [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置

输入: nums = [2,3,0,1,4]
输出: 2
```

### 解题思路
维护当前能到达的最远位置和下一步能到达的最远位置，当到达当前边界时跳跃次数加一。

### 代码实现
```python
def jump(nums):
    n = len(nums)
    if n <= 1:
        return 0
    
    jumps = 0
    current_end = 0  # 当前跳跃能到达的最远位置
    farthest = 0     # 下一步能到达的最远位置
    
    for i in range(n - 1):  # 不需要访问最后一个元素
        farthest = max(farthest, i + nums[i])
        if i == current_end:
            jumps += 1
            current_end = farthest
    
    return jumps
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

---

## 2. 划分字母区间（Partition Labels）

**题号**：763  
**难度**：中等

### 题目描述
给你一个字符串 `s`。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。

返回一个表示每个字符串片段的长度的列表。

### 示例
```
输入：s = "ababcbacadefegdehijhklij"
输出：[9,7,8]
解释：
划分结果为 "ababcbaca", "defegde", "hijhklij"
每个字母最多出现在一个片段中
像 "ababcbacadefegde", "hijhklij" 的划分是错误的，因为划分的片段数较少
```

### 解题思路
先记录每个字母最后出现的位置，然后贪心划分，确保当前片段包含所有字母的最后位置。

### 代码实现
```python
def partitionLabels(s):
    # 记录每个字母最后出现的位置
    last_occurrence = {char: idx for idx, char in enumerate(s)}
    
    result = []
    start = end = 0
    
    for i, char in enumerate(s):
        end = max(end, last_occurrence[char])
        if i == end:  # 当前片段结束
            result.append(end - start + 1)
            start = i + 1
    
    return result
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1) - 最多 26 个字母

---

## 3. 根据身高重建队列（Queue Reconstruction by Height）

**题号**：406  
**难度**：中等

### 题目描述
假设有打乱顺序的一群人站成一个队列，数组 `people` 表示队列中一些人的属性（不一定按顺序）。每个 `people[i] = [hi, ki]` 表示第 `i` 个人的身高为 `hi`，前面 **正好** 有 `ki` 个身高大于或等于 `hi` 的人。

请你重新构造并返回输入数组 `people` 所表示的队列。返回的队列应该格式化为数组 `queue`，其中 `queue[j] = [hj, kj]` 是队列中第 `j` 个人的属性（`queue[0]` 是排在队列前面的人）。

### 示例
```
输入：people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
输出：[[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
解释：
编号为 0 的人身高为 5 ，没有身高更高或者相同的人排在他前面
编号为 1 的人身高为 7 ，没有身高更高或者相同的人排在他前面
编号为 2 的人身高为 5 ，有 2 个身高更高或者相同的人排在他前面，即编号为 0 和 1 的人
编号为 3 的人身高为 6 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人
编号为 4 的人身高为 4 ，有 4 个身高更高或者相同的人排在他前面，即编号为 0、1、2、3 的人
编号为 5 的人身高为 7 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人
```

### 解题思路
先按身高降序、位置升序排序，然后按位置插入结果列表。

### 代码实现
```python
def reconstructQueue(people):
    # 按身高降序，位置升序排序
    people.sort(key=lambda x: (-x[0], x[1]))
    
    result = []
    for p in people:
        result.insert(p[1], p)  # 按位置插入
    
    return result
```

### 复杂度分析
- **时间复杂度**：O(n²) - insert 操作
- **空间复杂度**：O(n)

---

## 4. 用最少数量的箭引爆气球（Minimum Number of Arrows to Burst Balloons）

**题号**：452  
**难度**：中等

### 题目描述
有一些球形气球贴在一堵用 XY 平面表示的墙面上。墙面上的气球记录在整数数组 `points`，其中 `points[i] = [xstart, xend]` 表示水平直径在 `xstart` 和 `xend` 之间的气球。你不知道气球的确切 y 坐标。

一支弓箭可以沿着 x 轴从不同点 **完全垂直** 地射出。在坐标 `x` 处射出一支箭，若有一个气球的直径的开始和结束坐标为 `xstart`，`xend`，且满足 `xstart <= x <= xend`，则该气球会被 **引爆**。可以射出的弓箭的数量 **没有限制**。弓箭一旦被射出之后，可以无限地前进。

给你一个数组 `points`，返回引爆所有气球所必须射出的 **最小** 弓箭数。

### 示例
```
输入：points = [[10,16],[2,8],[1,6],[7,12]]
输出：2
解释：气球可以用 2 支箭来爆破:
- 在 x = 6 处射出箭，击破气球 [2,8] 和 [1,6]
- 在 x = 10 处射出箭，击破气球 [10,16] 和 [7,12]
```

### 解题思路
按结束位置排序，每次选择当前气球结束位置射箭，能射爆所有与之重叠的气球。

### 代码实现
```python
def findMinArrowShots(points):
    if not points:
        return 0
    
    # 按结束位置排序
    points.sort(key=lambda x: x[1])
    
    arrows = 1
    pos = points[0][1]  # 第一支箭的位置
    
    for i in range(1, len(points)):
        if points[i][0] > pos:  # 当前气球不能被射爆
            arrows += 1
            pos = points[i][1]  # 新的箭位置
    
    return arrows
```

### 复杂度分析
- **时间复杂度**：O(n·logn)
- **空间复杂度**：O(1)

---

## 5. 买卖股票的最佳时机 II（Best Time to Buy and Sell Stock II）

**题号**：122  
**难度**：中等

### 题目描述
给你一个整数数组 `prices`，其中 `prices[i]` 表示某支股票第 `i` 天的价格。

在每一天，你可以决定是否购买和/或出售股票。你在任何时候 **最多** 只能持有 **一股** 股票。你也可以先购买，然后在 **同一天** 出售。

返回你能获得的 **最大** 利润。

### 示例
```
输入：prices = [7,1,5,3,6,4]
输出：7
解释：在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5 - 1 = 4
随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6 - 3 = 3
总利润为 4 + 3 = 7

输入：prices = [1,2,3,4,5]
输出：4
解释：在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5 - 1 = 4
总利润为 4
```

### 解题思路
贪心，只要今天价格比昨天高，就在昨天买今天卖。

### 代码实现
```python
def maxProfit(prices):
    profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            profit += prices[i] - prices[i - 1]
    return profit
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)
