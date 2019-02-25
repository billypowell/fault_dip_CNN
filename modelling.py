import numpy as np

#############################################################################
#model building functions

def build_layercake():
    
    '''Build a layercake model 2d section'''
    
    #Generate 10 random value layers in a single trace
    rdm = np.random.rand(10,1)
    a = np.ones((10,1))

    for i in range(10):
        a = np.hstack((a, rdm))
        
    a = a[:,1:]
    final = np.ndarray.flatten(a)
    final = np.append(final,1)
    
    #Create section from trace & set values around 0
    b = []

    for i in range(100):
        t = final
        b.append(t)
    
    b = np.stack(b, axis=1)
    
    b = b-0.5
    return b



def build_synthetic(model, wavelet):
    
    '''Create reflectivity series from layers and convolve with a wavelet'''
    
    rc = np.diff(model, axis=0)
    
    synthetic = []

    for i in range(100):
    
        rt = rc[:,i]
    
        syn = np.convolve(rt, wavelet, mode='same')
    
        synthetic.append(syn)
    
    synthetic = np.stack(synthetic, axis=1)
    
    return synthetic



def build_fold(model,amplitude,frequency, posneg):
    
    shift_range = np.arange(100)

    shift_value = (amplitude * np.sin(shift_range * frequency)) * posneg

    shift_value = shift_value.astype(int)
    
    
    fold_syn = []

    for i in range(100):
    
        tr = model[:,i]
    
        troll = np.roll(tr, shift_value[i])
    
        fold_syn.append(troll)
    
    fold_syn = np.stack(fold_syn,axis=1)
    
    return fold_syn


#############################################################################
#faulting

def fault(model, kernel, displacement):
    
    hanging = np.roll(a=model, shift=displacement, axis=0)
    
    faulth = hanging * kernel
    
    faultf = model * np.flip(kernel)
    
    return faultf + faulth


#############################################################################
#training daa generation
def synthetic_generation(fault, wavelet, num_samples):

    tests = []

    for i in tqdm(range(num_samples)):
    
        randpole = rd.randint(0,1)

        if randpole == 1:
            pole = -1
        else:
            pole = 1
    
        flat_model = md.build_layercake()
        folded_syn = md.build_fold(flat_model,rd.randint(1,20),rd.uniform(0.05,0.1), posneg=pole)
        faulted = md.fault(folded_syn, fault, rd.randint(1,100))
        synthetic = md.build_synthetic(faulted, wavelet=wavelet)
    
    
        noise = br.noise.noise_db(synthetic, 15)
        synthetic = synthetic + noise
        cut  = synthetic[25:73,33:65]
        tests.append(cut)
    return np.stack(tests, axis=2)