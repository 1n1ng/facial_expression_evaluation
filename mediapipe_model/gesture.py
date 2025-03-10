import pickle

with open('fitting_params.pkl', 'rb') as f:
    data = pickle.load(f)
print(data["betas"])
print(data["betas"].shape   )

# import numpy as np

# aaa = np.load("lmk_3d.npz")
# print(aaa["lmk"].shape)