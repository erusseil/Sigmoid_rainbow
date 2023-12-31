import numpy as np

min_total_points = 6
PLASTICC_TARGET = {'Ia': 90, 'II': 42, 'Ibc': 62, 'SLSN': 95, 'KN': 64, 'TDE': 15,
                   'YSE_SNII':'YSE_SNII', 'YSE_SNIa':'YSE_SNIa', 'YSE_SNIbc':'YSE_SNIbc'}
PLASTICC_TARGET_INV = {90: 'Ia', 42: 'II', 62: 'Ibc', 95: 'SLSN', 64: 'KN', 15:'TDE',
                      'YSE_SNII':'YSE_SNII', 'YSE_SNIa':'YSE_SNIa', 'YSE_SNIbc':'YSE_SNIbc'}
PASSBANDS = np.array(['u ', 'g ', 'r ', 'i ', 'z ', 'Y '])