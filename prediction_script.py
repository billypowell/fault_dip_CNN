
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.preprocessing import normalize

from keras.utils import to_categorical
from keras.models import load_model

######################################################################################

def normalise(patches):

normalised_patches = []
patch_shape = patches.shape

for i in tqdm(range(patch_shape[0])):

patch = patches[i, :, :]
test = normalize(X=patch)
test = np.reshape(test, (48,32,1))

normalised_patches.append(test)

return np.stack(normalised_patches, axis=0)


def result_vector(array):

def decode(datum):
return np.argmax(datum)

res = []
for i in range(array.shape[0]):
datum = array[i]
decoded_datum = decode(array[i])
res.append(decoded_datum)

return res

######################################################################################

print('loading data...')

data = np.genfromtxt('16004_XL9804.dat')
data = data.T
data_subset = data[200:600,:]
del data

print('generating patches...')
patches = extract_patches_2d(image=data_subset, patch_size=(48,32))
print(str(patches.shape))


print('normalising patches...')
normalised_patch_stack = normalise(patches)
del patches

######################################################################################

print('loading CNN...')
model = load_model('faultCNNmodel.h5')

print('predicting:')
prediction = model.predict(normalised_patch_stack, verbose=1)

print('vectorising results...')
result = result_vector(prediction)
resstack = np.stack(result, axis=0)

print('saving results...')
np.save(arr=resstack, file='results_16004_XL9804_200600.npy')


print('Complete')




import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.preprocessing import normalize

from keras.utils import to_categorical
from keras.models import load_model

######################################################################################

def normalise(patches):

    normalised_patches = []
    patch_shape = patches.shape

    for i in tqdm(range(patch_shape[0])):
    
        patch = patches[i, :, :]
        test = normalize(X=patch)
        test = np.reshape(test, (48,32,1))
    
        normalised_patches.append(test)
        
    return np.stack(normalised_patches, axis=0)


def result_vector(array):

    def decode(datum):
        return np.argmax(datum)

    res = []
    for i in range(array.shape[0]):
        datum = array[i]
        decoded_datum = decode(array[i])
        res.append(decoded_datum)

    return res

######################################################################################

print('loading data...')

data = np.genfromtxt('16004_XL9804.dat')
data = data.T
data_subset = data[200:600,:]
del data

print('generating patches...')
patches = extract_patches_2d(image=data_subset, patch_size=(48,32))
print(str(patches.shape))


print('normalising patches...')
normalised_patch_stack = normalise(patches)
del patches

######################################################################################

print('loading CNN...')
model = load_model('faultCNNmodel.h5')

print('predicting:')
prediction = model.predict(normalised_patch_stack, verbose=1)

print('vectorising results...')
result = result_vector(prediction)
resstack = np.stack(result, axis=0)

print('saving results...')
np.save(arr=resstack, file='results_16004_XL9804_200600.npy')


print('Complete')




