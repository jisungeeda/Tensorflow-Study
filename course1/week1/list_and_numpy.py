#%%
import numpy as np

#%%
### LIST
filter = [ [0, -1, 0], [1, -4, 1], [0, 1, -2] ]
print(filter)
print(len(filter))
print(filter[0][1])
### NP array
filter_nd = np.array(filter)
print(filter_nd)
print(filter_nd.shape)
print(filter_nd[0][1])
print(filter_nd[0, 1])

### Takeaway ###
# List 는 .shape 가 없고 Numpy array 는 .shape 로 tuple 형태의 사이즈 정보를 얻을 수 있다.
# List 와 Numpy array 에 len() 으로 첫 번째 axis 의 사이즈를 알 수 있다.
# List 와 Numpy array 모두, 요소에 접근할 때에 [idx1][idx2] 이렇게 사용할 수 있다. 하지만 Numpy array 만 [idx1, idx2] 형태로 사용할 수 있다.
