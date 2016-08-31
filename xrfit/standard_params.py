


dog = """
linear = Enum(REPR='linear',
              pars=['slope', 'intercept'],
              
              
pvoigt = Enum(REPR='pvoigt',
              pars=['center', 'amplitude', 'sigma', 'fraction'],
              



pvoigt = {'center' : (None, True, 0.0, 0.0, None),
          'amplitude' : (None, True, 0.0, 0.0, None),
          'sigma' : (None, True, 0.0, 0.0, None),
          'fraction' : (None, True, 0.0, 0.0, None)}
          

uparams = {'bkg_slope': (-0.2, True, None, None, None),
           'bkg_intercept' : (2.0, False, 0, None, None),
           'peak_fraction' : (0.5, True, None, None, None),
           'peak_center' : (3.8, True, 3.2, 4.0, None),
           'peak_amplitude' : (6, True, 0, np.inf, None),
           'peak_sigma' : (0.35, True, 0.001, 3.0, None)
           }



peak01_sigma:       0.11226839 +/- 0.005802 (5.17%) (init= 0.08400001)
    peak01_center:      5.97392908 +/- 0.003305 (0.06%) (init= 5.966)
    peak01_fraction:    1.15710896 +/- 0.100485 (8.68%) (init= 0.5)
    peak01_amplitude:
"""
