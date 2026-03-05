# 滑动窗口（Sliding Window）

---

## 1. 无重复字符的最长子串（Longest Substring Without Repeating Characters）

**题号**：003  
**难度**：中等

### 题目描述
给定一个字符串 `s`，请你找出其中不含有重复字符的 **最长子串** 的长度。

### 示例
```
输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3

输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1
```

### 解题思路
双指针 + 哈希集合，右指针扩展窗口，遇到重复字符时左指针收缩，直到该字符不在重复。

### 代码实现
```python
def lengthOfLongestSubstring(s):
    char_set = set()
    left = max_len = 0
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)
    return max_len
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(min(m, n))，m 是字符集大小

---

## 2. 找到字符串中所有字母异位词（Find All Anagrams in a String）

**题号**：438  
**难度**：中等

### 题目描述
给定两个字符串 `s` 和 `p`，找到 `s` 中所有 `p` 的 **异位词** 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

**异位词** 指由相同字母以不同顺序组成的字符串。

### 示例
```
输入: s = "cbaebabacd", p = "abc"
输出: [0,6]
解释:
起始索引等于 0 的子串是 "cba"，是 "abc" 的异位词
起始索引等于 6 的子串是 "bac"，是 "abc" 的异位词
```

### 解题思路
固定窗口大小，滑动比较字符计数。

### 代码实现
```python
def findAnagrams(self, s: str, p: str) -> List[int]:
    m, n = len(s), len(p)
    if n > m: return []
    need = Counter(p)           # p 的字符需求
    window = Counter(s[:n])     # 初始窗口
    ans = []

    if window == need: 
        ans.append(0)
    for i in range(n, m):
        window[s[i]] += 1
        window[s[i - n]] -= 1
        if window[s[i - n]] == 0:
            del window[s[i - n]]
        
        if window == need:
            ans.append(i - n + 1)
    return ans
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)（字符集固定26个）

---

## 3. 和为 K 的子数组（Subarray Sum Equals K）

**题号**：560  
**难度**：中等

### 题目描述
给你一个整数数组 `nums` 和一个整数 `k`，请你统计并返回该数组中和为 `k` 的连续子数组的个数。

### 示例
```
输入：nums = [1,1,1], k = 2
输出：2
解释：有两个子数组和为 2：[1,1]（索引 0-1）和 [1,1]（索引 1-2）
```

### 解题思路
前缀和 + 哈希表，`prefix[j] - prefix[i] = k`，即找之前有多少个前缀和等于 `prefix[j] - k`。

### 代码实现
```python
def subarraySum(nums, k):
    from collections import defaultdict
    prefix_sum = defaultdict(int)
    prefix_sum[0] = 1
    curr_sum = count = 0
    for num in nums:
        curr_sum += num
        count += prefix_sum[curr_sum - k]
        prefix_sum[curr_sum] += 1
    return count
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(n)

---

## 4. 滑动窗口最大值（Sliding Window Maximum）

**题号**：239  
**难度**：困难

### 题目描述
给你一个整数数组 `nums`，有一个大小为 `k` 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 `k` 个数字。滑动窗口每次只向右移动一位。

返回滑动窗口中的最大值。

### 示例
```
输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
解释：
窗口位置                最大值
---------------        -----
[1  3  -1] -3  5  3  6  7   3
 1 [3  -1  -3] 5  3  6  7   3
 1  3 [-1  -3  5] 3  6  7   5
 1  3  -1 [-3  5  3] 6  7   5
 1  3  -1  -3 [5  3  6] 7   6
 1  3  -1  -3  5 [3  6  7]  7
```

### 解题思路
单调队列，队首维护最大值，移除窗口外元素。

### 代码实现
```python
def maxSlidingWindow(nums, k):
    from collections import deque
    q = deque()  # 存储索引
    res = []
    for i, num in enumerate(nums):
        while q and nums[q[-1]] < num:
            q.pop()
        q.append(i)
        if q[0] <= i - k:
            q.popleft()
        if i >= k - 1:
            res.append(nums[q[0]])
    return res
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(k)

### 解题思路
优先队列（堆），大根堆维护窗口元素和索引，根据索引决定是否弹出堆顶元素。

### 代码实现
```python
def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
    heap = []
    ans = []
    for i in range(len(nums)):
        heapq.heappush(heap, (-nums[i], i))
        if i >= k-1:
            while heap[0][1] <= i-k:
                heapq.heappop(heap)
            ans.append(-heap[0][0])
    return ans
```

### 复杂度分析
- **时间复杂度**：O(n log n)
- **空间复杂度**：O(n)

---

## 5. 最小覆盖子串（Minimum Window Substring）

**题号**：076  
**难度**：困难

### 题目描述
给你一个字符串 `s`、一个字符串 `t`。返回 `s` 中涵盖 `t` 所有字符的最小子串。如果不存在符合条件的子串，则返回空字符串 `""`。

**注意**：
- 对于 `t` 中重复字符，我们寻找的子字符串中该字符数量必须不少于 `t` 中该字符数量
- 如果 `s` 中存在这样的子串，我们保证它是唯一的答案

### 示例
```
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
解释：最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'
```

### 解题思路
双指针 + 字符计数，收缩左边界找最小窗口，难点在于窗口字符串判断。

### 代码实现
```python
def minWindow(s, t):
    from collections import Counter
    need = Counter(t)
    missing = len(t)
    left = start = 0
    min_len = float('inf')
    
    for right, c in enumerate(s, 1):
        if need[c] > 0:
            missing -= 1
        need[c] -= 1
        
        while missing == 0:
            if right - left < min_len:
                min_len = right - left
                start = left
            need[s[left]] += 1
            if need[s[left]] > 0:
                missing += 1
            left += 1
    
    return s[start:start + min_len] if min_len != float('inf') else ""
```

### 复杂度分析
- **时间复杂度**：O(|s| + |t|)
- **空间复杂度**：O(|t|)
