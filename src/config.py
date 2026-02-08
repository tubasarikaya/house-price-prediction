import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 17
TEST_SIZE = 0.20
QUANTILE_LOW = 0.10
QUANTILE_UP = 0.90
CORR_THRESHOLD = 0.70
RARE_PERC = 0.01

NO_FILL_COLS = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", 
    "BsmtFinType1", "BsmtFinType2", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", 
    "GarageCond", "PoolQC", "Fence", "MiscFeature"
]

DROP_COLS = [
    "Street", "Alley", "LandContour", "Utilities", "LandSlope",
    "Heating", "PoolQC", "MiscFeature", "Neighborhood"
]

CAT_LENGTH_THRESHOLD = 17
