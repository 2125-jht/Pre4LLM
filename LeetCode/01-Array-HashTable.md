# 数组与哈希表（Array & HashTable）

---

## 1. 两数之和（Two Sum）

**题号**：001  
**难度**：简单

### 题目描述
给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** 的那 **两个** 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

### 示例
```
输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9，返回 [0, 1]
```

### 解题思路
使用哈希表存储已遍历的数字及其索引。遍历数组时，检查 `target - nums[i]` 是否在哈希表中。

### 代码实现
```python
def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
```

### 复杂度分析
- **时间复杂度**：O(n) - 只需遍历数组一次
- **空间复杂度**：O(n) - 哈希表存储 n 个元素

---

## 2. 字母异位词分组（Group Anagrams）

**题号**：049  
**难度**：中等

### 题目描述
给你一个字符串数组，请你将 **字母异位词** 组合在一起。可以按任意顺序返回结果列表。

**字母异位词** 是由重新排列源单词的字母得到的一个新单词，所有源单词中的字母通常恰好只用一次。

### 示例
```
输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
```

### 解题思路
排序后的字符串作为 key，哈希表分组。

### 代码实现
```python
def groupAnagrams(strs):
    from collections import defaultdict
    groups = defaultdict(list)
    for s in strs:
        key = ''.join(sorted(s))
        groups[key].append(s)
    return list(groups.values())
```

### 复杂度分析
- **时间复杂度**：O(n·k·logk) - n 是字符串数量，k 是字符串最大长度
- **空间复杂度**：O(n·k)

---

## 3. 最长连续序列（Longest Consecutive Sequence）

**题号**：128  
**难度**：中等

### 题目描述
给定一个未排序的整数数组 `nums`，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现时间复杂度为 O(n) 的算法解决此问题。

### 示例
```
输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]，长度为 4
```

### 解题思路
哈希集合，只从序列起点开始计数（即 num-1 不在集合中的数）。

### 代码实现
```python
def longestConsecutive(nums):
    num_set = set(nums)
    longest = 0
    for num in num_set:
        if num - 1 not in num_set:  # 只从起点开始
            curr, streak = num, 1
            while curr + 1 in num_set:
                curr += 1
                streak += 1
            longest = max(longest, streak)
    return longest
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(n)

---

## 4. 移动零（Move Zeroes）

**题号**：283  
**难度**：简单

### 题目描述
给定一个数组 `nums`，编写一个函数将所有 `0` 移动到数组的末尾，同时保持非零元素的相对顺序。

**请注意**，必须在不复制数组的情况下原地对数组进行操作。

### 示例
```
输入: nums = [0,1,0,3,12]
输出: [1,3,12,0,0]
```

### 解题思路
双指针，非零元素前移，后面补零。

### 代码实现
```python
def moveZeroes(nums):
    j = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[j], nums[i] = nums[i], nums[j]
            j += 1
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

---

## 5. 盛最多水的容器（Container With Most Water）

**题号**：011  
**难度**：中等

### 题目描述
给定一个长度为 `n` 的整数数组 `height`。有 `n` 条垂线，第 `i` 条线的两个端点是 `(i, 0)` 和 `(i, height[i])`。

找出其中的两条线，使得它们与 `x` 轴共同构成的容器可以容纳最多的水。返回容器可以储存的最大水量。

**说明**：你不能倾斜容器。

### 示例
```
输入：[1,8,6,2,5,4,8,3,7]
输出：49
解释：选择第 1 条线（height=8）和第 8 条线（height=7），面积为 min(8,7) * (8-1) = 49
```

### 解题思路
双指针，移动较短的一边，因为面积受限于较短边。

### 代码实现
```python
def maxArea(height):
    left, right = 0, len(height) - 1
    max_area = 0
    while left < right:
        width = right - left
        if height[left] < height[right]:
            max_area = max(max_area, height[left] * width)
            left += 1
        else:
            max_area = max(max_area, height[right] * width)
            right -= 1
    return max_area
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

---

## 6. 三数之和（3Sum）

**题号**：015  
**难度**：中等

### 题目描述
给你一个整数数组 `nums`，判断是否存在三元组 `[nums[i], nums[j], nums[k]]` 满足 `i != j`、`i != k` 且 `j != k`，同时还满足 `nums[i] + nums[j] + nums[k] == 0`。

请你返回所有和为 `0` 且不重复的三元组。

**注意**：答案中不可以包含重复的三元组。

### 示例
```
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
```

### 解题思路
排序 + 双指针，去重是关键。

### 代码实现
```python
def threeSum(nums):
    nums.sort()
    res = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0: left += 1
            elif total > 0: right -= 1
            else:
                res.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left+1]: left += 1
                while left < right and nums[right] == nums[right-1]: right -= 1
                left += 1; right -= 1
    return res
```

### 复杂度分析
- **时间复杂度**：O(n²)
- **空间复杂度**：O(1)（不计结果）

---

## 7. 接雨水（Trapping Rain Water）

**题号**：042  
**难度**：困难

### 题目描述
给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

### 示例
```
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：可以接 6 个单位的雨水
```

### 解题思路
双指针，分别从两端出发，维护左右最大高度，谁那边低就先处理谁，计算当前位置能接的雨水量。

### 代码实现
```python
def trap(self, height: List[int]) -> int:
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    ans = 0
    
    while left < right:
        if height[left] < height[right]:
            # 左边矮：右边一定有个更高的墙挡着，安全接水
            if height[left] >= left_max:
                left_max = height[left]  # 更新左边最高墙
            else:
                ans += left_max - height[left]  # 接水！
            left += 1
        else:
            # 右边矮：左边一定有个更高的墙挡着，安全接水  
            if height[right] >= right_max:
                right_max = height[right]
            else:
                ans += right_max - height[right]
            right -= 1
    return ans
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

### 解题思路
单调栈，想象你不是一格一格算水，而是一层一层铺水。栈里存的是"可能成为左边边界的墙"（索引），保持从栈底到栈顶递减。当遇到一堵更高的墙时，栈顶那堵墙就成了凹槽底部，可以计算它和新的墙、以及栈中下一个墙形成的"水坑"。

### 代码实现
```python
def trap(self, height: List[int]) -> int:
    stack = []  # 存储索引，维护递减栈
    ans = 0
    
    for i, h in enumerate(height):
        while stack and h > height[stack[-1]]:
            top = stack.pop()
            if not stack:
                break
            # 计算当前层接水量
            distance = i - stack[-1] - 1
            bounded_height = min(height[stack[-1]], h) - height[top]
            ans += distance * bounded_height
        stack.append(i)
    return ans
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(n)

---

## 8. 除自身以外数组的乘积（Product of Array Except Self）

**题号**：238  
**难度**：中等

### 题目描述
给你一个整数数组 `nums`，返回 **数组 `answer`** ，其中 `answer[i]` 等于 `nums` 中除 `nums[i]` 之外其余各元素的乘积。

题目数据 **保证** 数组 `nums`之中任意元素的全部前缀元素和后缀的乘积都在 **32 位** 整数范围内。

请**不要使用除法**，且在 `O(n)` 时间复杂度内完成此题。

### 示例
```
输入: nums = [1,2,3,4]
输出: [24,12,8,6]
解释：
- answer[0] = 2*3*4 = 24
- answer[1] = 1*3*4 = 12
- answer[2] = 1*2*4 = 8
- answer[3] = 1*2*3 = 6

输入: nums = [-1,1,0,-3,3]
输出: [0,0,9,0,0]
```

### 解题思路
两次遍历，先计算左边所有元素的乘积，再计算右边所有元素的乘积。

### 代码实现
```python
def productExceptSelf(nums):
    n = len(nums)
    answer = [1] * n
    
    # 先计算左边所有元素的乘积
    left_product = 1
    for i in range(n):
        answer[i] = left_product
        left_product *= nums[i]
    
    # 再计算右边所有元素的乘积
    right_product = 1
    for i in range(n - 1, -1, -1):
        answer[i] *= right_product
        right_product *= nums[i]
    
    return answer
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)（输出数组不计入空间复杂度）

---

## 9. 寻找重复数（Find the Duplicate Number）

**题号**：287  
**难度**：中等

### 题目描述
给定一个包含 `n + 1` 个整数的数组 `nums`，其数字都在 `[1, n]` 范围内（包括 `1` 和 `n`），可知至少存在一个重复的整数。

假设 `nums` 只有 **一个重复的整数**，返回这个重复的数。

你设计的解决方案必须 **不修改** 数组 `nums` 且只用常量级 `O(1)` 的额外空间。

### 示例
```
输入：nums = [1,3,4,2,2]
输出：2

输入：nums = [3,1,3,4,2]
输出：3
```

### 解题思路
 Floyd 判圈算法（快慢指针），将数组看作链表，重复数字即环的入口。

### 代码实现
```python
def findDuplicate(nums):
    # 快慢指针找环
    slow = fast = nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    
    # 找环的入口
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    
    return slow
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

---

## 10. 缺失的第一个正数（First Missing Positive）

**题号**：041  
**难度**：困难

### 题目描述
给你一个未排序的整数数组 `nums`，请你找出其中没有出现的最小的正整数。

请你实现时间复杂度为 `O(n)` 并且只使用常数级别额外空间的解决方案。

### 示例
```
输入：nums = [1,2,0]
输出：3
解释：数字 1, 2 存在，下一个正整数是 3

输入：nums = [3,4,-1,1]
输出：2

输入：nums = [7,8,9,11,12]
输出：1
```

### 解题思路
原地哈希，将数字 `i` 放到索引 `i-1` 处，然后遍历找第一个不匹配的位置。

### 代码实现
```python
def firstMissingPositive(nums):
    n = len(nums)
    
    # 原地哈希：将数字 i 放到索引 i-1 处
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
    
    # 找第一个不匹配的位置
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    
    return n + 1
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

---

## 11. 轮转数组（Rotate Array）

**题号**：189  
**难度**：中等

### 题目描述
给定一个整数数组 `nums`，将数组中的元素向右轮转 `k` 个位置，其中 `k` 是非负数。

### 示例
```
输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右轮转 1 步: [7,1,2,3,4,5,6]
向右轮转 2 步: [6,7,1,2,3,4,5]
向右轮转 3 步: [5,6,7,1,2,3,4]
```

### 解题思路
三次反转：先整体反转，再分别反转前 k 个和剩余部分。

### 代码实现
```python
def rotate(nums, k):
    n = len(nums)
    k %= n  # 处理 k > n 的情况
    
    def reverse(left, right):
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
    
    reverse(0, n - 1)      # 整体反转
    reverse(0, k - 1)      # 反转前 k 个
    reverse(k, n - 1)      # 反转剩余部分
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

---

## 12. 最大子数组和（Maximum Subarray）

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

## 13. 合并区间（Merge Intervals）

**题号**：056  
**难度**：中等

### 题目描述
以数组 `intervals` 表示若干个区间的集合，其中单个区间为 `intervals[i] = [starti, endi]` 。请你合并所有重叠的区间，并返回 **一个不重叠的区间数组**，该数组需恰好覆盖输入中的所有区间。

### 示例
```
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6]

输入：intervals = [[1,4],[4,5]]
输出：[[1,5]]
解释：区间 [1,4] 和 [4,5] 可被视为重叠区间
```

### 解题思路
先按区间的 **起点** 进行排序，遍历排序后的区间，如果当前区间起点 <= 上一个区间的终点，说明有重叠，合并（更新终点为两者最大值）

### 代码实现
```python
def merge(intervals):
    if not intervals:
        return []
    
    # 按区间起点排序
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]  # 结果列表，先放入第一个区间
    
    for i in range(1, len(intervals)):
        curr = intervals[i]
        last = merged[-1]
        
        if curr[0] <= last[1]:  # 有重叠，合并
            last[1] = max(last[1], curr[1])
        else:  # 无重叠，加入新区间
            merged.append(curr)
    
    return merged
```

### 复杂度分析
- **时间复杂度**：O(n·logn) - 主要是排序的时间复杂度
- **空间复杂度**：O(n) - 存储合并后的结果，或 O(logn) 到 O(n) 的排序栈空间
