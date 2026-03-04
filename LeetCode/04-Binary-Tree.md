# 二叉树（Binary Tree）

---

## 1. 二叉树的中序遍历（Binary Tree Inorder Traversal）

**题号**：094  
**难度**：简单

### 题目描述
给定一个二叉树的根节点 `root`，返回它的 **中序** 遍历。

### 示例
```
输入：root = [1,null,2,3]
输出：[1,3,2]
解释：
  1
   \
    2
   /
  3
```

### 解题思路
递归或栈迭代，左-根-右。

### 代码实现
```python
def inorderTraversal(root):
    res, stack = [], []
    curr = root
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        res.append(curr.val)
        curr = curr.right
    return res
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(n)

---

## 2. 二叉树的最大深度（Maximum Depth of Binary Tree）

**题号**：104  
**难度**：简单

### 题目描述
给定一个二叉树，找出其最大深度。

二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

### 示例
```
输入：root = [3,9,20,null,null,15,7]
输出：3
解释：
    3
   / \
  9  20
    /  \
   15   7
```

### 解题思路
递归，深度 = max(左子树深度, 右子树深度) + 1。

### 代码实现
```python
def maxDepth(root):
    if not root:
        return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(h)，h 为树高

---

## 3. 翻转二叉树（Invert Binary Tree）

**题号**：226  
**难度**：简单

### 题目描述
给你一棵二叉树的根节点 `root`，翻转这棵二叉树，并返回其根节点。

### 示例
```
输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]
解释：
     4              4
   /   \          /   \
  2     7   →    7     2
 / \   / \      / \   / \
1   3 6   9    9   6 3   1
```

### 解题思路
递归交换左右子树。

### 代码实现
```python
def invertTree(root):
    if not root:
        return None
    root.left, root.right = invertTree(root.right), invertTree(root.left)
    return root
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(h)

---

## 4. 二叉树的直径（Diameter of Binary Tree）

**题号**：543  
**难度**：简单

### 题目描述
给你一棵二叉树的根节点，返回该树的 **直径** 。

二叉树的 **直径** 是指树中任意两个节点之间最长路径的 **长度** 。这条路径可能经过也可能不经过根节点 `root` 。

两节点之间路径的 **长度** 由它们之间边数表示。

### 示例
```
输入：root = [1,2,3,4,5]
输出：3
解释：3 为直径的长度，即 [4,2,1,3] 或 [5,2,1,3] 这条路径
```

### 解题思路
后序遍历，直径 = 左深度 + 右深度。

### 代码实现
```python
def diameterOfBinaryTree(root):
    self.diameter = 0
    def depth(node):
        if not node:
            return 0
        left = depth(node.left)
        right = depth(node.right)
        self.diameter = max(self.diameter, left + right)
        return 1 + max(left, right)
    depth(root)
    return self.diameter
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(h)

---

## 5. 二叉树的层序遍历（Binary Tree Level Order Traversal）

**题号**：102  
**难度**：中等

### 题目描述
给你二叉树的根节点 `root`，返回其节点值的 **层序遍历**。（即逐层地，从左到右访问所有节点）。

### 示例
```
输入：root = [3,9,20,null,null,15,7]
输出：[[3],[9,20],[15,7]]
```

### 解题思路
BFS，队列逐层处理。

### 代码实现
```python
def levelOrder(root):
    if not root:
        return []
    from collections import deque
    queue, res = deque([root]), []
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        res.append(level)
    return res
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(n)

---

## 6. 将有序数组转换为二叉搜索树（Convert Sorted Array to BST）

**题号**：108  
**难度**：简单

### 题目描述
给你一个整数数组 `nums`，其中元素已经按 **升序** 排列，请你将其转换为一棵 **高度平衡** 二叉搜索树。

**高度平衡** 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1」的二叉树。

### 示例
```
输入：nums = [-10,-3,0,5,9]
输出：[0,-3,9,-10,null,5]
解释：[0,-10,5,null,-3,null,9] 也是正确的
```

### 解题思路
二分递归，中间元素作为根。

### 代码实现
```python
def sortedArrayToBST(nums):
    if not nums:
        return None
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = sortedArrayToBST(nums[:mid])
    root.right = sortedArrayToBST(nums[mid+1:])
    return root
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(log n)

---

## 7. 验证二叉搜索树（Validate Binary Search Tree）

**题号**：098  
**难度**：中等

### 题目描述
给你一个二叉树的根节点 `root`，判断其是否是一个有效的二叉搜索树。

**有效** 二叉搜索树定义如下：
- 节点的左子树只包含 **小于** 当前节点的数
- 节点的右子树只包含 **大于** 当前节点的数
- 所有左子树和右子树自身必须也是二叉搜索树

### 示例
```
输入：root = [2,1,3]
输出：true

输入：root = [5,1,4,null,null,3,6]
输出：false
解释：根节点的值是 5，但是右子节点的值是 4
```

### 解题思路
中序遍历应为递增序列，或递归判断范围。

### 代码实现
```python
def isValidBST(root, min_val=float('-inf'), max_val=float('inf')):
    if not root:
        return True
    if root.val <= min_val or root.val >= max_val:
        return False
    return (isValidBST(root.left, min_val, root.val) and
            isValidBST(root.right, root.val, max_val))
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(h)

---

## 8. 二叉树展开为链表（Flatten Binary Tree to Linked List）

**题号**：114  
**难度**：中等

### 题目描述
给你二叉树的根结点 `root`，请你将它展开为一个单链表：
- 展开后的单链表应该同样使用 `TreeNode`，其中 `right` 子指针指向链表中下一个结点，而左子指针始终为 `null`
- 展开后的单链表应该与二叉树 **先序遍历** 顺序相同

### 示例
```
输入：root = [1,2,5,3,4,null,6]
输出：[1,null,2,null,3,null,4,null,5,null,6]
```

### 解题思路
递归，将左右子树分别展开后拼接。

### 代码实现
```python
def flatten(root):
    if not root:
        return
    flatten(root.left)
    flatten(root.right)
    
    temp = root.right
    root.right = root.left
    root.left = None
    
    curr = root
    while curr.right:
        curr = curr.right
    curr.right = temp
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(h)
