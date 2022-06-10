





# 同时对数据进行筛选（filter）和计算 （sum , min， max）
# exp1：
nums = [range(1,10)]
s = sum(x*x for x in nums) ### generator 
#### : s = sum([x*x for x in nums]) list comprehension 引入了额外的列表，只使用一次就丢弃，基于生成器的解决方案可以以迭代的方式转换数据，在内存上高效很多。
# exp2：
import os
files = os.listdir('dirname')
if any(name.endswith('.py') for name in files):
    print('There be python!')

# exp3: tuple to csv
s = ('ACME', 50, 123.45)
print(','.join(str(x) for x in s))

#exp4: data reduction 
portfolio = [
    {'name':'GOOG', 'shares':50},
    {'name':'YHOO', 'shares':70},
    {'name':'AOL', 'shares':20},
    {'name':'SCOX', 'shares':65},
]
##return 20
min_shares = min(s['shares'] for s in portfolio)
##{'name':'AOL', 'shares':20},
min_shares = min(portfolio, key=lambda s : s['shares'])
from operator import itemgetter
key = itemgetter('shares')
 


# ChainMap: 将多个映射合并