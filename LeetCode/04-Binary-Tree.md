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

## 8. 二叉搜索树中第 K 小的元素（Kth Smallest Element in a BST）

**题号**：230  
**难度**：中等

### 题目描述
给定一个二叉搜索树的根节点 `root`，和整数 `k`，请你设计一个算法查找其中第 `k` 小的元素。

### 示例
```
输入：root = [3,1,4,null,2], k = 1
输出：1

输入：root = [5,3,6,2,4,null,null,1], k = 3
输出：3
```

### 解题思路
二叉搜索树中序遍历为递增序列，第 k 个元素即为第 k 小的元素。

### 代码实现
```python
def kthSmallest(root, k):
    stack = []
    curr = root
    
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        k -= 1
        if k == 0:
            return curr.val
        curr = curr.right
```

### 复杂度分析
- **时间复杂度**：O(H + k)，H 为树高，平均 O(log n + k)，最坏 O(n + k)
- **空间复杂度**：O(H)，递归栈空间

---

## 9. 二叉树展开为链表（Flatten Binary Tree to Linked List）

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
先序遍历过程中，用 `pre` 指针记录前一个访问的节点，将 `pre.right` 指向当前节点，`pre.left` 置为 `None`，实现原地展开。

### 代码实现
```python
def flatten(root):
    self.pre = None
    def preorder(node):
        if not node:
            return
        # 保存左右子树引用
        left, right = node.left, node.right
        # 将 pre 的右指针指向当前节点
        if self.pre:
            self.pre.left = None
            self.pre.right = node
        self.pre = node
        # 递归处理左右子树
        preorder(left)
        preorder(right)
    preorder(root)
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(h)

---

## 10. 对称二叉树（Symmetric Tree）

**题号**：101  
**难度**：简单

### 题目描述
给你一个二叉树的根节点 `root`，检查它是否轴对称。

### 示例
```
输入：root = [1,2,2,3,4,4,3]
输出：true

输入：root = [1,2,2,null,3,null,3]
输出：false
```

### 解题思路
递归比较左右子树是否镜像对称。

### 代码实现
```python
def isSymmetric(root):
    if not root:
        return True
    
    def isMirror(left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        return (left.val == right.val and
                isMirror(left.left, right.right) and
                isMirror(left.right, right.left))
    
    return isMirror(root.left, root.right)
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(h)

---

## 11. 从前序与中序遍历序列构造二叉树（Construct BT from Preorder and Inorder）

**题号**：105  
**难度**：中等

### 题目描述
给定两个整数数组 `preorder` 和 `inorder`，其中 `preorder` 是二叉树的**先序遍历**，`inorder` 是同一棵树的**中序遍历**，请构造二叉树并返回其根节点。

### 示例
```
输入: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
输出: [3,9,20,null,null,15,7]
```

### 解题思路
先序第一个元素是根，在中序中找到根的位置划分左右子树。

### 代码实现
```python
def buildTree(preorder, inorder):
    if not preorder or not inorder:
        return None
    
    # 先序第一个元素是根
    root_val = preorder[0]
    root = TreeNode(root_val)
    
    # 在中序中找到根的位置
    root_idx = inorder.index(root_val)
    
    # 递归构建左右子树
    root.left = buildTree(preorder[1:1+root_idx], inorder[:root_idx])
    root.right = buildTree(preorder[1+root_idx:], inorder[root_idx+1:])
    
    return root
```

### 复杂度分析
- **时间复杂度**：O(n²) - 可用哈希表优化到 O(n)
- **空间复杂度**：O(n)

---

## 12. 路径总和 III（Path Sum III）

**题号**：437  
**难度**：中等

### 题目描述
给定一个二叉树的根节点 `root`，和一个整数目标值 `targetSum`，求该二叉树里节点值之和等于 `targetSum` 的 **路径** 的数目。

**路径** 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

### 示例
```
输入：root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
输出：3
解释：和等于 8 的路径有:
1. 5 -> 3
2. 5 -> 2 -> 1
3. -3 -> 11
```

### 解题思路
前缀和 + 哈希表，类似数组中和为 k 的子数组个数。

### 代码实现
```python
def pathSum(root, targetSum):
    from collections import defaultdict
    prefix_sum = defaultdict(int)
    prefix_sum[0] = 1  # 前缀和为 0 的路径有 1 条
    
    def dfs(node, curr_sum):
        if not node:
            return 0
        
        curr_sum += node.val
        count = prefix_sum[curr_sum - targetSum]  # 以当前节点结尾的路径数
        
        prefix_sum[curr_sum] += 1
        count += dfs(node.left, curr_sum)
        count += dfs(node.right, curr_sum)
        prefix_sum[curr_sum] -= 1  # 回溯
        
        return count
    
    return dfs(root, 0)
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(n)

---

## 13. 二叉树的右视图（Binary Tree Right Side View）

**题号**：199  
**难度**：中等

### 题目描述
给定一个二叉树的 **根节点** `root`，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

### 示例
```
输入: [1,2,3,null,5,null,4]
输出: [1,3,4]
解释:
       1            <---
     /   \
    2     3         <---
     \     \
      5     4       <---
```

### 解题思路
BFS 层序遍历，每层最后一个节点；或 DFS 优先遍历右子树。

### 代码实现
```python
def rightSideView(root):
    if not root:
        return []
    from collections import deque
    queue = deque([root])
    res = []
    
    while queue:
        level_size = len(queue)
        for i in range(level_size):
            node = queue.popleft()
            if i == level_size - 1:  # 每层最后一个节点
                res.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return res
```

### DFS 解法

```python
def rightSideView(root):
    res = []
    
    def dfs(node, depth):
        if not node:
            return
        # 第一次访问该深度时，即为右视图看到的节点
        if depth == len(res):
            res.append(node.val)
        # 优先遍历右子树，确保右视图节点先被访问
        dfs(node.right, depth + 1)
        dfs(node.left, depth + 1)
    
    dfs(root, 0)
    return res
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(n)

---

## 14. 二叉树中的最大路径和（Binary Tree Maximum Path Sum）

**题号**：124  
**难度**：困难

### 题目描述
二叉树中的 **路径** 被定义为一条节点序列，序列中每对相邻节点之间都存在一条边。同一个节点在一条路径序列中 **至多出现一次** 。该路径 **至少包含一个** 节点，且不一定经过根节点。

**路径和** 是路径中各节点值的总和。

给你一个二叉树的根节点 `root`，返回其 **最大路径和** 。

### 示例
```
输入：root = [1,2,3]
输出：6
解释：最优路径是 2 -> 1 -> 3 ，路径和为 2 + 1 + 3 = 6

输入：root = [-10,9,20,null,null,15,7]
输出：42
解释：最优路径是 15 -> 20 -> 7 ，路径和为 15 + 20 + 7 = 42
```

### 解题思路
后序遍历，计算以当前节点为起点的最大路径和，更新全局最大值。

### 代码实现
```python
def maxPathSum(root):
    self.max_sum = float('-inf')
    
    def maxGain(node):
        if not node:
            return 0
        
        # 递归计算左右子树的最大贡献值（负数取0表示不选）
        left_gain = max(maxGain(node.left), 0)
        right_gain = max(maxGain(node.right), 0)
        
        # 以当前节点为根的路径和
        price_newpath = node.val + left_gain + right_gain
        self.max_sum = max(self.max_sum, price_newpath)
        
        # 返回以当前节点为起点的最大路径和（只能选一边）
        return node.val + max(left_gain, right_gain)
    
    maxGain(root)
    return self.max_sum
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(h)
