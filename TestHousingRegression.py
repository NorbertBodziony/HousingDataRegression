import pandas as pd
import numpy as np
import HousingRegression as hr

from unittest import TestCase
from sklearn.datasets import load_boston

class TestHousingRegression(TestCase):

    global k, boston_data, target_cut
    boston_market_data = load_boston()
    boston_data = pd.DataFrame(boston_market_data.data, columns=[boston_market_data.feature_names])
    k = boston_data.values
    k = k.T
    target_cut = np.array(boston_market_data['target'])

    def test_to_delete_min_function(self):
        all_min = {0, 41, 161, 365, 450}
        to_delete_min = set()
        hr.to_delete_min_function(k, to_delete_min)
        assert all_min == to_delete_min

    def test_to_delete_max_function(self):
        all_max = {0, 1, 4, 7, 8, 11, 13, 22, 23, 31, 35, 37, 45, 46, 48, 49, 52, 53, 54, 56, 59, 62, 65, 66, 68, 69,
                   80, 82, 84, 88, 94, 97, 99, 113, 128, 129, 131, 139, 141, 142, 143, 144, 145, 146, 151, 158, 159,
                   172, 179, 181, 183, 189, 196, 199, 205, 209, 218, 223, 253, 258, 259, 271, 274, 275, 278, 279, 287,
                   288, 290, 291, 292, 293, 294, 295, 297, 306, 307, 308, 311, 317, 320, 321, 322, 324, 326, 327, 335,
                   336, 339, 340, 343, 350, 364, 367, 368, 371, 373, 374, 375, 377, 378, 379, 380, 381, 382, 383, 385,
                   386, 387, 388, 389, 392, 393, 394, 396, 398, 400, 401, 402, 403, 405, 406, 407, 409, 410, 411, 412,
                   413, 414, 415, 418, 420, 437, 439, 442, 443, 448, 459, 462, 464, 469, 470, 480, 492, 493, 494, 496,
                   497, 498, 500, 502, 503, 505}
        to_delete_max = set()
        hr.to_delete_max_function(k, to_delete_max)
        assert all_max == to_delete_max

    def test_amount_to_stay_max(self):
        all_max = {0, 1, 4, 7, 8, 11, 13, 22, 23, 31, 35, 37, 45, 46, 48, 49, 52, 53, 54, 56, 59, 62, 65, 66, 68, 69,
                   80, 82, 84, 88, 94, 97, 99, 113, 128, 129, 131, 139, 141, 142, 143, 144, 145, 146, 151, 158, 159,
                   172, 179, 181, 183, 189, 196, 199, 205, 209, 218, 223, 253, 258, 259, 271, 274, 275, 278, 279, 287,
                   288, 290, 291, 292, 293, 294, 295, 297, 306, 307, 308, 311, 317, 320, 321, 322, 324, 326, 327, 335,
                   336, 339, 340, 343, 350, 364, 367, 368, 371, 373, 374, 375, 377, 378, 379, 380, 381, 382, 383, 385,
                   386, 387, 388, 389, 392, 393, 394, 396, 398, 400, 401, 402, 403, 405, 406, 407, 409, 410, 411, 412,
                   413, 414, 415, 418, 420, 437, 439, 442, 443, 448, 459, 462, 464, 469, 470, 480, 492, 493, 494, 496,
                   497, 498, 500, 502, 503, 505}
        amount = 353
        target_cut_new = hr.drop_max_elements_from_boston_data(boston_data, all_max, target_cut)
        assert target_cut_new.shape[0] == amount

    def test_amount_to_stay_min(self):
        all_min = {0, 41, 161, 365, 450}
        amount = 503
        target_cut_new = hr.drop_min_elements_from_boston_data(boston_data, all_min, target_cut)
        assert target_cut_new.shape[0] == amount

