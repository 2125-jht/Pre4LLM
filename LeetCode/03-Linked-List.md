# 链表（Linked List）

---

## 1. 两数相加（Add Two Numbers）

**题号**：002  
**难度**：中等

### 题目描述
给你两个 **非空** 的链表，表示两个非负的整数。它们每位数字都是按照 **逆序** 的方式存储的，并且每个节点只能存储 **一位** 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

### 示例
```
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807

输入：l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
输出：[8,9,9,9,0,0,0,1]
```

### 解题思路
模拟加法，注意进位和链表长度不等。

### 代码实现
```python
def addTwoNumbers(l1, l2):
    dummy = ListNode(0)
    curr, carry = dummy, 0
    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        total = val1 + val2 + carry
        carry, val = divmod(total, 10)
        curr.next = ListNode(val)
        curr = curr.next
        if l1: l1 = l1.next
        if l2: l2 = l2.next
    return dummy.next
```

### 复杂度分析
- **时间复杂度**：O(max(m,n))
- **空间复杂度**：O(max(m,n))

---

## 2. 合并两个有序链表（Merge Two Sorted Lists）

**题号**：021  
**难度**：简单

### 题目描述
将两个升序链表合并为一个新的 **升序** 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

### 示例
```
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
```

### 解题思路
递归或迭代，每次选较小节点。

### 代码实现
```python
def mergeTwoLists(l1, l2):
    dummy = ListNode(0)
    curr = dummy
    while l1 and l2:
        if l1.val < l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    curr.next = l1 or l2
    return dummy.next
```

### 复杂度分析
- **时间复杂度**：O(m+n)
- **空间复杂度**：O(1)

---

## 3. 反转链表（Reverse Linked List）

**题号**：206  
**难度**：简单

### 题目描述
给你单链表的头节点 `head`，请你反转链表，并返回反转后的链表。

### 示例
```
输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]

输入：head = [1,2]
输出：[2,1]
```

### 解题思路
三指针迭代，逐个反转指向。

### 代码实现
```python
def reverseList(head):
    prev, curr = None, head
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    return prev
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

---

## 4. 删除链表的倒数第 N 个结点（Remove Nth Node From End）

**题号**：019  
**难度**：中等

### 题目描述
给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。

### 示例
```
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]

输入：head = [1], n = 1
输出：[]
```

### 解题思路
快慢指针，先让快指针走 N 步。

### 代码实现
```python
def removeNthFromEnd(head, n):
    dummy = ListNode(0, head)
    fast = slow = dummy
    for _ in range(n + 1):
        fast = fast.next
    while fast:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return dummy.next
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

---

## 5. 两两交换链表中的节点（Swap Nodes in Pairs）

**题号**：024  
**难度**：中等

### 题目描述
给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。

### 示例
```
输入：head = [1,2,3,4]
输出：[2,1,4,3]

输入：head = []
输出：[]

输入：head = [1]
输出：[1]
```

### 解题思路
使用虚拟头节点，每次处理一对节点，注意指针交换的顺序。

### 代码实现
```python
def swapPairs(head):
    dummy = ListNode(0, head)
    prev = dummy
    
    while prev.next and prev.next.next:
        # 定位两个要交换的节点
        first = prev.next
        second = prev.next.next
        
        # 交换
        prev.next = second
        first.next = second.next
        second.next = first
        
        # 移动prev到下一对节点之前
        prev = first
    
    return dummy.next
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

---

## 6. K 个一组翻转链表（Reverse Nodes in k-Group）

**题号**：025  
**难度**：困难

### 题目描述
给你链表的头节点 `head` ，每 `k` 个节点一组进行翻转，请你返回修改后的链表。

`k` 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 `k` 的整数倍，那么请将最后剩余的节点保持原有顺序。

你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

### 示例
```
输入：head = [1,2,3,4,5], k = 2
输出：[2,1,4,3,5]

输入：head = [1,2,3,4,5], k = 3
输出：[3,2,1,4,5]
```

### 解题思路
1. 先遍历链表，记录每 k 个分组的起始节点
2. 对每个完整的 k 个节点分组进行反转
3. 将反转后的分组连接起来，剩余不足 k 个的节点保持原顺序

### 代码实现
```python
def reverseKLink(head, k):
    pre = None
    cur = head
    while k > 0:
        k -= 1
        tmp = cur.next
        cur.next = pre
        pre = cur
        cur = tmp
    return pre

def reverseKGroup(head, k):
    head_Nodes = []
    i = 0
    cur = head
    while cur:
        if i % k == 0:
            head_Nodes.append(cur)
        cur = cur.next
        i += 1
    
    dummy = ListNode()
    pre = dummy
    if i % k == 0:
        # 恰好是 k 的整数倍
        for i in range(len(head_Nodes)):
            pre.next = reverseKLink(head_Nodes[i], k)
            pre = head_Nodes[i]
    else:
        # 剩余节点不足 k 个，保持原顺序
        for i in range(len(head_Nodes) - 1):
            pre.next = reverseKLink(head_Nodes[i], k)
            pre = head_Nodes[i]
        pre.next = head_Nodes[-1]
    
    return dummy.next
```

### 复杂度分析
- **时间复杂度**：O(n)，每个节点访问常数次
- **空间复杂度**：O(n/k)，用于存储每组的起始节点

---

## 7. 合并 K 个升序链表（Merge k Sorted Lists）

**题号**：023  
**难度**：困难

### 题目描述
给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

### 示例
```
输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到：
1->1->2->3->4->4->5->6
```

### 解题思路
优先队列（堆）或分治合并。

### 代码实现
```python
def mergeKLists(lists):
    import heapq
    dummy = ListNode(0)
    curr = dummy
    heap = []
    for i, l in enumerate(lists):
        if l:
            heapq.heappush(heap, (l.val, i, l))
    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next
```

### 复杂度分析
- **时间复杂度**：O(N·logk)，N 是所有节点总数
- **空间复杂度**：O(k)

### 另一种 O(1) 空间解法（逐轮找最小值）
```python
def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    if not lists:
        return None
    
    k = len(lists)
    dummy = ListNode(0)
    cur = dummy
    while True:
        min_idx = -1
        min_val = float('inf')
        # 找当前最小值的链表索引
        for i in range(k):
            if lists[i] and lists[i].val < min_val:
                min_val = lists[i].val
                min_idx = i
        if min_idx == -1:  # 所有链表都为空
            break
        # 直接连接原节点
        cur.next = lists[min_idx]
        cur = cur.next
        lists[min_idx] = lists[min_idx].next

    return dummy.next
```
- **时间复杂度**：O(k·N)，每找一个节点都要遍历 k 个链表的头节点
- **空间复杂度**：O(1)

---

## 8. 环形链表Ⅱ（Linked List Cycle II）

**题号**：142  
**难度**：中等

### 题目描述
给定一个链表的头节点 `head`，返回链表开始入环的第一个节点。如果链表无环，则返回 `null`。

如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。

### 示例
```
输入：head = [3,2,0,-4], pos = 1
输出：tail connects to node index 1
解释：链表中有一个环，其尾部连接到第二个节点
```

### 解题思路
快慢指针：
1. 先判断是否有环（快慢指针相遇）
2. 有环时，将慢指针移回头部，两指针同速前进，再次相遇点即为环入口

数学推导：设头到环入口距离为 a，环入口到相遇点距离为 b，相遇点到环入口距离为 c。快指针走了 a+b+c+b，慢指针走了 a+b。由于快指针速度是慢指针2倍：2(a+b) = a+b+c+b，可得 a = c。

### 代码实现
```python
def detectCycle(head):
    slow = fast = head
    
    # 第一步：判断是否有环，找到相遇点
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # 无环
    
    # 第二步：找环入口
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

---

## 9. 相交链表（Intersection of Two Linked Lists）

**题号**：160  
**难度**：简单

### 题目描述
给你两个单链表的头节点 `headA` 和 `headB`，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 `null`。

### 示例
```
输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5]
输出：Intersected at '8'
解释：相交节点的值为 8（注意，如果两个链表相交则不能为 0）
```

### 解题思路
双指针交替遍历，消除长度差，数学证明两指针走的总路程相同。

### 代码实现
```python
def getIntersectionNode(headA, headB):
    pA, pB = headA, headB
    while pA != pB:
        pA = pA.next if pA else headB
        pB = pB.next if pB else headA
    return pA
```

### 复杂度分析
- **时间复杂度**：O(m+n)
- **空间复杂度**：O(1)

---

## 10. 排序链表（Sort List）

**题号**：148  
**难度**：中等

### 题目描述
给你链表的头结点 `head`，请将其按 **升序** 排列并返回 **排序后的链表**。

### 示例
```
输入：head = [4,2,1,3]
输出：[1,2,3,4]

输入：head = [-1,5,3,4,0]
输出：[-1,0,3,4,5]
```

### 解题思路
归并排序，快慢指针找中点，递归排序左右两部分后合并。

### 代码实现
```python
def sortList(head):
    if not head or not head.next:
        return head
    
    # 快慢指针找中点
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    mid = slow.next
    slow.next = None
    
    # 递归排序
    left = sortList(head)
    right = sortList(mid)
    
    # 合并两个有序链表
    dummy = ListNode(0)
    curr = dummy
    while left and right:
        if left.val < right.val:
            curr.next = left
            left = left.next
        else:
            curr.next = right
            right = right.next
        curr = curr.next
    curr.next = left or right
    
    return dummy.next
```

### 复杂度分析
- **时间复杂度**：O(n·logn)
- **空间复杂度**：O(logn) - 递归栈空间

---

## 11. 随机链表的复制（Copy List with Random Pointer）

**题号**：138  
**难度**：中等

### 题目描述
给你一个长度为 `n` 的链表，每个节点包含一个额外增加的随机指针 `random`，该指针可以指向链表中的任何节点或空节点。

构造这个链表的 **深拷贝**。深拷贝应该正好由 `n` 个 **全新** 节点组成，其中每个新节点的值与其对应的原节点的值相同。

### 示例
```
输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]
```

### 解题思路
哈希表存储原节点到新节点的映射，两次遍历完成复制。

### 代码实现
```python
def copyRandomList(head):
    if not head:
        return None
    
    # 第一次遍历：创建新节点并建立映射
    old_to_new = {}
    curr = head
    while curr:
        old_to_new[curr] = ListNode(curr.val)
        curr = curr.next
    
    # 第二次遍历：设置 next 和 random 指针
    curr = head
    while curr:
        if curr.next:
            old_to_new[curr].next = old_to_new[curr.next]
        if curr.random:
            old_to_new[curr].random = old_to_new[curr.random]
        curr = curr.next
    
    return old_to_new[head]
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(n)

---

## 12. 回文链表（Palindrome Linked List）

**题号**：234  
**难度**：简单

### 题目描述
给你一个单链表的头节点 `head`，请判断该链表是否为回文链表。如果是，返回 `true`；否则，返回 `false`。

### 示例
```
输入：head = [1,2,2,1]
输出：true

输入：head = [1,2]
输出：false
```

### 解题思路
快慢指针找中点，反转后半部分，然后比较前后两部分。

### 代码实现
```python
def isPalindrome(head):
    if not head or not head.next:
        return True
    
    # 快慢指针找中点
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    # 反转后半部分
    prev = None
    while slow:
        next_temp = slow.next
        slow.next = prev
        prev = slow
        slow = next_temp
    
    # 比较前后两部分
    left, right = head, prev
    while right:  # 后半部分可能更短
        if left.val != right.val:
            return False
        left = left.next
        right = right.next
    
    return True
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

---

## 13. 奇偶链表（Odd Even Linked List）

**题号**：328  
**难度**：中等

### 题目描述
给定单链表的头节点 `head`，将所有索引为奇数的节点和索引为偶数的节点分别组合在一起，然后返回重新排序的列表。

**第一个**节点的索引被认为是 **奇数**，**第二个**节点的索引为 **偶数**，以此类推。

请注意，偶数组和奇数组内部的相对顺序应该与输入时保持一致。

### 示例
```
输入: head = [1,2,3,4,5]
输出: [1,3,5,2,4]

输入: head = [2,1,3,5,6,4,7]
输出: [2,3,6,7,1,5,4]
```

### 解题思路
分离奇偶节点，然后将偶链表接到奇链表后面。

### 代码实现
```python
def oddEvenList(head):
    if not head or not head.next:
        return head
    
    odd = head           # 奇数索引节点头
    even = head.next     # 偶数索引节点头
    even_head = even     # 保存偶数头
    
    while even and even.next:
        odd.next = even.next
        odd = odd.next
        even.next = odd.next
        even = even.next
    
    odd.next = even_head
    return head
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

---

## 14. LRU 缓存（LRU Cache）

**题号**：146  
**难度**：中等

### 题目描述
请你设计并实现一个满足 **LRU (最近最少使用) 缓存** 约束的数据结构。

实现 `LRUCache` 类：
- `LRUCache(int capacity)` 以 **正整数** 作为容量 `capacity` 初始化 LRU 缓存
- `int get(int key)` 如果关键字 `key` 存在于缓存中，则返回关键字的值，否则返回 `-1`
- `void put(int key, int value)` 如果关键字 `key` 已经存在，则变更其数据值 `value`；如果不存在，则向缓存中插入该组 `key-value`。如果插入操作导致关键字数量超过 `capacity`，则应该 **逐出** 最久未使用的关键字

函数 `get` 和 `put` 必须以 `O(1)` 的平均时间复杂度运行。

### 示例
```
输入
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
输出
[null, null, null, 1, null, -1, null, -1, 3, 4]
```

### 解题思路
使用 **双向链表 + 哈希表**：
- **双向链表**：维护访问顺序，头部是最近使用的，尾部是最久未使用的
- **哈希表**：存储 key 到链表节点的映射，实现 O(1) 查找

### 代码实现
```python
class LinkNode:
    """双向链表节点"""
    def __init__(self, key=0, value=None):
        self.key = key
        self.value = value
        self.pre = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.map = {}
        self.head = LinkNode(-1, -1)
        self.tail = LinkNode(-1, -1)
        self.tail.pre = self.head
        self.head.next = self.tail

    def get(self, key: int) -> int:
        if key in self.map:
            node = self.map[key]
            self.move2head(node)
            return node.value
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.map:
            node = self.map[key]
            node.value = value
            self.move2head(node)
        else:
            cur = len(self.map)
            node = LinkNode(key, value)
            if cur == self.capacity:
                self.pop(self.tail.pre)
                self.push(node)
            else:
                self.push(node)

    def push(self, node: LinkNode):
        """将节点添加到头部（最近使用）"""
        a = self.head.next
        self.head.next = node
        node.pre = self.head
        node.next, a.pre = a, node
        self.map[node.key] = node

    def pop(self, node):
        """删除指定节点"""
        a, b = node.pre, node.next
        a.next, b.pre = b, a
        del self.map[node.key]

    def move2head(self, node):
        """将节点移动到头部"""
        self.pop(node)
        self.push(node)
```

### 复杂度分析
- **时间复杂度**：`get` 和 `put` 都是 O(1)
- **空间复杂度**：O(capacity)，哈希表和双向链表最多存储 capacity + 2 个节点
