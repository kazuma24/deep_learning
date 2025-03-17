import matplotlib.pyplot as plt
import numpy as np

def step_function( x ):
      
    return np.array( x > 0, dtype=np.int64 )


x = np.arange(-5.0, 5.0, 0.1)
print(type(x))
y= step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # Y軸の範囲
plt.show()