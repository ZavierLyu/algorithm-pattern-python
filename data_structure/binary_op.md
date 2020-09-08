# 二进制

## 常见二进制操作

### 基本操作

a=0^a=a^0

0=a^a

由上面两个推导出：a=a^b^b

### 交换两个数

a=a^b

b=a^b

a=a^b

### 移除最后一个 1

a=n&(n-1)
去掉最右边为1的位数（将其转换成0）

```python
0b100010 & (0b100001) = 0b100000
```

### 获取最后一个 1

最小的1保存，其他为0
diff=n&(-n)

```python
0b100010 & 0b011110 = 0b000010
```

## 常见题目

### [single-number](https://leetcode-cn.com/problems/single-number/)

> 给定一个**非空**整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

```Python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        
        out = 0
        for num in nums:
            out ^= num
        
        return out
```

### [single-number-ii](https://leetcode-cn.com/problems/single-number-ii/)

> 给定一个**非空**整数数组，除了某个元素只出现一次以外，其余每个元素均出现了三次。找出那个只出现了一次的元素。

这里给这题讲解一下，根据leetcode[TankCode](https://leetcode-cn.com/u/tankcode/)大佬的题解进行自己分析的。

首先我们要定义重复3次的状态，由于一个bit只能记录重复2次的状态，所以我们需要2个bit来记录，也就是`00,01,10` 分别对应着重复1次，重复2次和重复3次。我们用XY来分别表明这2个bit，然后我们需要记录在不同阶段，当输入为1和0时，这2个bit有什么样的变换。看表：

| XY   | Z    | $X_{new}$ | $Y_{new}$ |
| ---- | ---- | --------- | --------- |
| 00   | 0    | 0         | 0         |
| 01   | 0    | 0         | 1         |
| 10   | 0    | 1         | 0         |
| 00   | 1    | 0         | 1         |
| 01   | 1    | 1         | 0         |
| 10   | 1    | 0         | 0         |

如此我们就记录了针对各种情况下XY的相应变换。针对此题，我们是要去掉重复3次的数，也就是说最好能实现类似`x ^ x=0`的情况，所以我们定义重复3次的情况为`10`, 出现1次的情况为`01`, 此时我们只需要取Y就能拿到只出现1次的数。然后我们只要列$Y_{new}$ 和 $X_{new}$的递推公式就可以了，学过逻辑电路的都可以很快得到


$$
Y_{new}= \overline X Y \overline Z + \overline X \overline Y Z = \overline X (Y\oplus Z) \\
X_{new} = X \overline Y \overline Z + \overline X Y Z \\
= \overline Y_{new} (X\oplus Z)
$$
为什么$X_{new}$ 可以用到$Y_{new}$, 因为我们已经拿到$Y_{new}$后，可以根据第四列，第二列和第一列的X值来推断了，这样就不需要额外的变量来存储$Y_{new}$了，两种推断公式本质上是一样的。 


```Python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        X = Y = 0
        for Z in nums:
            Y = ~X & (Y ^ Z)
            X = ~Y & (X ^ Z)
        return Y
```

### [single-number-iii](https://leetcode-cn.com/problems/single-number-iii/)

> 给定一个整数数组  `nums`，其中恰好有两个元素只出现一次，其余所有元素均出现两次。 找出只出现一次的那两个元素。

基于官方解答给点解释。我们的目标是要找到X,Y这两个只出现了一次的数。我们用之前的方法其实可以得到`X^Y`。所以问题的关键在于如何从`X^Y`区分`X`和`Y`。

首先我们知道`X`和`Y`并不相同，所以`X^Y`必定存在一个位（这里我们用`n & (-n)`来找到最右的1）为1，记为第k位，既然`Xk^Yk`=1, 那么就必定存在`Xk=1,Yk=0`或者是`Xk=0,Yk=1`。WLOG, 我们假定`Xk=1,Yk=0`，那么我们只要去nums里面找到这个第k位为1的且不重复的数即可。然后`Y=X^Y^X`。
```Python
class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        X = Y = 0
        XxorY = 0
        for i in nums:
            XxorY ^= i

        keyMask = XxorY & (-XxorY)

        for i in nums:
            if keyMask & i: # 首先得是第k位为1
                X ^= i      # 其次是去重复

        Y = XxorY ^ X 
        return [X, Y]
```

### [number-of-1-bits](https://leetcode-cn.com/problems/number-of-1-bits/)

> 编写一个函数，输入是一个无符号整数，返回其二进制表达式中数字位数为 ‘1’  的个数（也被称为[汉明重量](https://baike.baidu.com/item/%E6%B1%89%E6%98%8E%E9%87%8D%E9%87%8F)）。

```Python
class Solution:
    def hammingWeight(self, n: int) -> int:
        num_ones = 0
        while n > 0:
            num_ones += 1
            n &= n - 1
        return num_ones
```

### [counting-bits](https://leetcode-cn.com/problems/counting-bits/)

> 给定一个非负整数  **num**。对于  0 ≤ i ≤ num  范围中的每个数字  i ，计算其二进制数中的 1 的数目并将它们作为数组返回。

- 思路：利用上一题的解法容易想到 O(nk) 的解法，k 为位数。但是实际上可以利用动态规划将复杂度降到 O(n)，想法其实也很简单，即当前数的 1 个数等于比它少一个 1 的数的结果加 1。下面给出三种 DP 解法

```Python
# x <- x // 2
class Solution:
    def countBits(self, num: int) -> List[int]:
        
        num_ones = [0] * (num + 1)
        
        for i in range(1, num + 1):
            num_ones[i] = num_ones[i >> 1] + (i & 1) # 注意位运算的优先级
        
        return num_ones
```

```Python
# x <- x minus right most 1
class Solution:
    def countBits(self, num: int) -> List[int]:
        
        num_ones = [0] * (num + 1)
        
        for i in range(1, num + 1):
            num_ones[i] = num_ones[i & (i - 1)] + 1
        
        return num_ones
```

```Python
# x <- x minus left most 1
class Solution:
    def countBits(self, num: int) -> List[int]:
        
        num_ones = [0] * (num + 1)
        
        left_most = 1
        
        while left_most <= num:
            for i in range(left_most):
                if i + left_most > num:
                    break
                num_ones[i + left_most] = num_ones[i] + 1
            left_most <<= 1
        
        return num_ones
```

### [reverse-bits](https://leetcode-cn.com/problems/reverse-bits/)

> 颠倒给定的 32 位无符号整数的二进制位。

思路：简单想法依次颠倒即可。更高级的想法是考虑到处理超长比特串时可能出现重复的pattern，此时如果使用 cache 记录出现过的 pattern 并在重复出现时直接调用结果可以节约时间复杂度，具体可以参考 leetcode 给出的解法。

```Python
import functools

class Solution:
    def reverseBits(self, n):
        ret, power = 0, 24
        while n:
            ret += self.reverseByte(n & 0xff) << power
            n = n >> 8
            power -= 8
        return ret

    # memoization with decorator
    @functools.lru_cache(maxsize=256)
    def reverseByte(self, byte):
        return (byte * 0x0202020202 & 0x010884422010) % 1023
```

### [bitwise-and-of-numbers-range](https://leetcode-cn.com/problems/bitwise-and-of-numbers-range/)

> 给定范围 [m, n]，其中 0 <= m <= n <= 2147483647，返回此范围内所有数字的按位与（包含 m, n 两端点）。

思路：直接从 m 到 n 遍历一遍显然不是最优。一个性质，如果 m 不等于 n，则结果第一位一定是 0 （中间必定包含一个偶数）。利用这个性质，类似的将 m 和 n 右移后我们也可以判断第三位、第四位等等，免去了遍历的时间复杂度。

真正的思路是 m 到 n, 在其公共前缀之外，必定会在每一位上都出现至少1次0，这些0可以 在与计算中给直接转换为0.所以我们只要找到公共前缀即可。
```Python
class Solution:
    def rangeBitwiseAnd(self, m: int, n: int) -> int:
        
        shift = 0
        while m < n:
            shift += 1
            m >>= 1
            n >>= 1
        
        return m << shift
```

## 练习

- [x] [single-number](https://leetcode-cn.com/problems/single-number/)
- [x] [single-number-ii](https://leetcode-cn.com/problems/single-number-ii/)
- [x] [single-number-iii](https://leetcode-cn.com/problems/single-number-iii/)
- [x] [number-of-1-bits](https://leetcode-cn.com/problems/number-of-1-bits/)
- [x] [counting-bits](https://leetcode-cn.com/problems/counting-bits/)
- [x] [reverse-bits](https://leetcode-cn.com/problems/reverse-bits/)
