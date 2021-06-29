import numpy as np
class shipModel:
    @staticmethod
    def speedGPS2Water(v, heading_ship, currentU, currentV):
        '''
        Function to convert speed over ground into speed through water
        '''
        V_water = v - currentU * np.sin(np.deg2rad(heading_ship)) - currentV * np.cos(np.deg2rad(heading_ship))

        return V_water