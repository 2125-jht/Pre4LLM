# 栈与堆（Stack & Heap）

---

## 1. 有效的括号（Valid Parentheses）

**题号**：020  
**难度**：简单

### 题目描述
给定一个只包括 `'('`，`')'`，`'{'`，`'}'`，`'['`，`']'` 的字符串 `s`，判断字符串是否有效。

有效字符串需满足：
1. 左括号必须用相同类型的右括号闭合
2. 左括号必须以正确的顺序闭合
3. 每个右括号都有一个对应的相同类型的左括号

### 示例
```
输入：s = "()"
输出：true

输入：s = "()[]{}"
输出：true

输入：s = "(]"
输出：false

输入：s = "([)]"
输出：false

输入：s = "{[]}"
输出：true
```

### 解题思路
栈，遇到左括号入栈，右括号检查栈顶是否匹配。

### 代码实现
```python
def isValid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping:
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
        else:
            stack.append(char)
    return not stack
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(n)

---

## 2. 每日温度（Daily Temperatures）

**题号**：739  
**难度**：中等

### 题目描述
给定一个整数数组 `temperatures`，表示每天的温度，返回一个数组 `answer`，其中 `answer[i]` 是指对于第 `i` 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 `0` 来代替。

### 示例
```
输入: temperatures = [73,74,75,71,69,72,76,73]
输出: [1,1,4,2,1,1,0,0]
解释：
第 0 天 73°F，第 1 天 74°F（更高），所以 answer[0] = 1
第 1 天 74°F，第 2 天 75°F（更高），所以 answer[1] = 1
第 2 天 75°F，第 6 天 76°F（更高），所以 answer[2] = 4
...
第 6 天 76°F，之后没有更高温度，所以 answer[6] = 0
```

### 解题思路
单调递减栈，找下一个更大元素。

### 代码实现
```python
def dailyTemperatures(temperatures):
    stack = []  # 存储索引
    res = [0] * len(temperatures)
    for i, temp in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temp:
            prev_idx = stack.pop()
            res[prev_idx] = i - prev_idx
        stack.append(i)
    return res
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(n)

---

## 3. 柱状图中最大的矩形（Largest Rectangle in Histogram）

**题号**：084  
**难度**：困难

### 题目描述
给定 *n* 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

### 示例
```
输入：heights = [2,1,5,6,2,3]
输出：10
解释：最大的矩形为 5 和 6 围成的矩形，面积为 10

输入：heights = [2,4]
输出：4
```

### 解题思路
单调递增栈，找左右边界。

### 代码实现
```python
def largestRectangleArea(heights):
    stack = []
    max_area = 0
    heights = [0] + heights + [0]  # 添加哨兵
    
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    return max_area
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(n)

---

## 4. 数组中的第 K 个最大元素（Kth Largest Element in an Array）

**题号**：215  
**难度**：中等

### 题目描述
给定整数数组 `nums` 和整数 `k`，请返回数组中第 `k` 个最大的元素。

请注意，你需要找的是数组排序后的第 `k` 个最大的元素，而不是第 `k` 个不同的元素。

你必须设计并实现时间复杂度为 `O(n)` 的算法解决此问题。

### 示例
```
输入: [3,2,1,5,6,4], k = 2
输出: 5

输入: [3,2,3,1,2,4,5,5,6], k = 4
输出: 4
```

### 解题思路
最小堆（维护 k 个元素）或快速选择。

### 代码实现（最小堆）
```python
def findKthLargest(nums, k):
    import heapq
    heap = nums[:k]
    heapq.heapify(heap)
    for num in nums[k:]:
        if num > heap[0]:
            heapq.heapreplace(heap, num)
    return heap[0]
```

### 复杂度分析
- **时间复杂度**：O(n·log k)
- **空间复杂度**：O(k)

**快速选择**：平均 O(n)，最坏 O(n²)

---

## 5. 数据流的中位数（Find Median from Data Stream）

**题号**：295  
**难度**：困难

### 题目描述
**中位数**是有序整数列表中的中间值。如果列表的大小是偶数，则没有中间值，中位数是两个中间值的平均值。

例如 `[2,3,4]` 的中位数是 `3`，`[2,3]` 的中位数是 `(2 + 3) / 2 = 2.5`。

设计一个支持以下两种操作的数据结构：
- `void addNum(int num)` - 从数据流中添加一个整数到数据结构中
- `double findMedian()` - 返回目前所有元素的中位数

### 示例
```
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
输出：[null, null, null, 1.5, null, 2.0]
解释：
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1,2]
medianFinder.findMedian(); // 返回 1.5 (即 (1+2)/2)
medianFinder.addNum(3);    // arr = [1,2,3]
medianFinder.findMedian(); // 返回 2.0
```

### 解题思路
双堆，大顶堆存较小一半，小顶堆存较大一半。

### 代码实现
```python
import heapq

class MedianFinder:
    def __init__(self):
        self.small = []  # 大顶堆（存负数）
        self.large = []  # 小顶堆
    
    def addNum(self, num):
        if len(self.small) == len(self.large):
            heapq.heappush(self.large, -heapq.heappushpop(self.small, -num))
        else:
            heapq.heappush(self.small, -heapq.heappushpop(self.large, num))
    
    def findMedian(self):
        if len(self.small) == len(self.large):
            return (-self.small[0] + self.large[0]) / 2
        return self.large[0]
```

### 复杂度分析
- **addNum**：O(log n)
- **findMedian**：O(1)
- **空间复杂度**：O(n)
