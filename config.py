import numpy as np 

# camera
CaM = np.asarray([ [378.7, 0, 314.5], [0, 377.8, 246.4], [0, 0, 1] ])

focal = (CaM[0,0] + CaM[1,1])/2 * 1.1

TEST_DPP = np.asarray([CaM[:2,2]]).ravel()
#YAW_BIAS = 10/ 180* np.pi
YAW_BIAS = 0 

# SIFT
MIN_MATCH_COUNT = 8

SIFT_nfeatures = 500
SIFT_nOctaveLayers = 3 
SIFT_contrastThreshold = 0.009
SIFT_edgeThreshold = 8
SIFT_sigma = 0.5

NUM_LOWE_BEST = 128
NUM_HOMO_STEPS = 3

# track
NUM_OK_STEPS = 16
OUT_THRESH = 5