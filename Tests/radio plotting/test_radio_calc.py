import unittest
from radio_calc import Location
from scipy.spatial import distance

class TestRadioCalc(unittest.TestCase):
    def setUp(self):
        BS_NO = 20
        DU_NO = 4
        RU_PER_DU_NO = 5
        X_LIM = 1000
        PRB_NO = 100
        USER_NO = 5
        VELOCITY = 35
        RAYLEIGH_SCALE =1
        ETA_AREA = 3
        FH_BW_CAPACITY=5000
        E2_BW_CAPACITY=100000
        self.radio_calc = Location(BS_NO, DU_NO, RU_PER_DU_NO, PRB_NO, USER_NO, VELOCITY, X_LIM, RAYLEIGH_SCALE, ETA_AREA, FH_BW_CAPACITY, E2_BW_CAPACITY)  # Replace with the actual class name

    def test_ric_du_distance(self):
        # Set up test data
        self.radio_calc.mat_cu_ric_loc = [(500, 500)]  # Replace with actual values
        self.radio_calc.mat_du_loc = [(250, 250), (800, 800), (400,400), (100,50)]  # Replace with actual values

        # Call the method under test
        distances = self.radio_calc.ric_du_distance()

        # Assert the expected result
        expected_distances = [distance.euclidean(self.radio_calc.mat_cu_ric_loc), distance.euclidean(self.radio_calc.mat_du_loc)]
        print('s')
        self.assertEqual(distances, expected_distances)

if __name__ == '__main__':
    unittest.main()