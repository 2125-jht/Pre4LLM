# 栈与堆（Stack & Heap）

---

## 1. 有效的括号（Valid Parentheses）

**题号**：020  
**思路**：栈，遇到左括号入栈，右括号检查栈顶是否匹配  
**代码**：
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
**复杂度**：时间 O(n)，空间 O(n)

---

## 2. 每日温度（Daily Temperatures）

**题号**：739  
**思路**：单调递减栈，找下一个更大元素  
**代码**：
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
**复杂度**：时间 O(n)，空间 O(n)

---

## 3. 柱状图中最大的矩形（Largest Rectangle in Histogram）

**题号**：084  
**思路**：单调递增栈，找左右边界  
**代码**：
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
**复杂度**：时间 O(n)，空间 O(n)

---

## 4. 数组中的第 K 个最大元素（Kth Largest Element in an Array）

**题号**：215  
**思路**：最小堆（维护 k 个元素）或快速选择  
**代码**（最小堆）：
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
**复杂度**：时间 O(n·log k)，空间 O(k)  
**快速选择**：平均 O(n)，最坏 O(n²)

---

## 5. 数据流的中位数（Find Median from Data Stream）

**题号**：295  
**思路**：双堆，大顶堆存较小一半，小顶堆存较大一半  
**代码**：
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
**复杂度**：addNum O(log n)，findMedian O(1)，空间 O(n)
