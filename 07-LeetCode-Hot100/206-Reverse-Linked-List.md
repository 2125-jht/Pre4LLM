# 206. Reverse Linked List（反转链表）

## 题目描述

给你单链表的头节点 `head`，请你反转链表，并返回反转后的链表。

## 解题思路

迭代法：使用三个指针（prev, curr, next），依次反转每个节点的指向。

## 代码实现

```python
def reverseList(head):
    prev = None
    curr = head
    
    while curr:
        next_temp = curr.next  # 暂存下一个节点
        curr.next = prev       # 反转指向
        prev = curr            # prev 前移
        curr = next_temp       # curr 前移
    
    return prev  # 新的头节点
```

## 复杂度分析

- **时间复杂度**：O(n) - 遍历链表一次
- **空间复杂度**：O(1) - 只使用常数额外空间

## 关键点

- 保存 next 节点再反转，避免断链
- 最后返回 prev（原链表的尾节点）
- 递归法也可解，但空间复杂度为 O(n)
