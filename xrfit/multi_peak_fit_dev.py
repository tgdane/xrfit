#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import print_function

__author__ = "Thomas Dane"
__contact__ = "dane@esrf.fr"
__license__ = "GPLv3+"
__copyright__ = "ESRF - The European Synchrotron, Grenoble, France"
__date__ = "08/06/2015"
__status__ = "Development"

import sys
import numpy as np
import matplotlib.pyplot as plt
import lmfit.models as models

from scipy.ndimage import label
from scipy.ndimage.filters import gaussian_filter1d as gauss1d

import matplotlib
import matplotlib.pyplot as plt

try:
    import yaml
    has_yaml = True
except ImportError:
    print('yaml library not found. Models cannot be loaded in yaml format')
    has_yaml = False

font = {'family': 'sans-serif',
        'style': 'normal',
        'weight': 'normal',
        'size': 14}

matplotlib.rc('font', **font)


class MultiPeakFit(object):
    """
    """

    def __init__(self, sigma1=100, sigma2=5, offset=0,
                 bkg_prefix='bkg', peak_prefix='peak',
                 require_errors=True, error_thresh=1.0):
        """
        """
        self._fsigma1 = sigma1
        self._fsigma2 = sigma2
        self._foffset = offset

        self._bkg_prefix = bkg_prefix
        self._peak_prefix = peak_prefix
        self._n_bkg_funcs = 0
        self._n_peak_funcs = 0

        self._model = None
        self._param = None

        self._err_thresh = error_thresh

        # output results
        self.f1 = None
        self.f2 = None
        self.fltd = None
        self.lbls = None
        self.nps = None

        self.out = None  # lmfit fit result object - contains everything
        self.fit = None  # array of best fit data
        self.res = None  # array of residuals
        self.bkg = None  # array of background
        self.bkg_comp = None  # dictionary of bkg component arrays
        self.peaks = None  # dictionary of fit peak component arrays
        self.init = None  # array of initial guess data
        self.covar = None  # covariance matrix
        self.coeffs = None  # fit coefficients
        self.stderr = None  # standard error

    all_funcs = {'gauss': 'GaussianModel',
                 'lorentz': 'LorentzianModel',
                 'voigt': 'VoigtModel',
                 'pvoigt': 'PseudoVoigtModel',
                 'pearson7': 'Pearson7Model',
                 'lognormal': 'LognormalModel',
                 'expgauss': 'ExponentialGaussianModel',
                 'skewgauss': 'SkewedGaussianModel',
                 'doniach': 'DoniachModel',
                 'constant': 'ConstantModel',
                 'linear': 'LinearModel',
                 'quadratic': 'QuadraticModel',
                 'polynomial': 'PolynomialModel',
                 'exponential': 'ExponentialModel',
                 'powerlaw': 'PowerLawModel'}

    def reset(self):
        """
        Hard reset all model and parameters objects.
        """
        self._model = None
        self._param = None
        self._n_bkg_funcs = 0
        self._n_peak_funcs = 0
        self.reset_results()

    def reset_results(self):
        """
        Hard reset results.
        """
        self.out = None
        self.fit = None
        self.res = None
        self.bkg = None
        self.bkg_comp = None
        self.peaks = None
        self.init = None
        self.covar = None
        self.coeffs = None
        self.stderr = None

    def new_func(self, func, prefix, guess=False, y=None, x=None):
        """
        Generic method for getting a new model and parameters.
        
        Parameters
        ----------
        func : string
            Name of the function
        prefix : string
            Prefix to be appended to start of parameter names
        guess : bool
            Specificies whether or not to guess the parameter values.
            Data (y and x) are required if True
        y : array
            y data
        x : array
            x data
            
        Returns
        -------
        model : lmfit model
        param : lmfit parameters
        """
        if func not in self.all_funcs.keys():
            raise RuntimeError('Invalid function')
        method = getattr(models, self.all_funcs[func.lower()])
        model = method(prefix=prefix)

        if guess:
            if (y is None) or (x is None):
                raise RuntimeError(('MultiPeakFit.add_func error: Cannot guess'
                                    ' model parameters without data.'))
            param = model.guess(y, x=x)
        else:
            param = model.make_params()
        return model, param

    def add_bkg(self, func='linear', guess=False, y=None, x=None):
        """
        Wrapper for MultiPeakFit.add_func to add a background. 
        Default function is linear. Model and Parameter will be 
        stored in the class instance.
        
        Parameters
        ----------
        func : string
            Function name
        guess : bool
            Guess initial parameter values
        y : array
            y data
        x : array
            x data
        
        Returns
        -------
        prefix : string
            Prefix of the new model parameters. Can be useful for
            modifying parameters after creation.
        """
        prefix = '{}{:02}_'.format(self._bkg_prefix, self._n_bkg_funcs + 1)
        model, param = self.new_func(func, prefix, guess, y, x)
        print(model, param)
        if self._model is None:
            self._model = model
            self._param = param
        else:
            self._model += model
            self._param += param
        self._n_bkg_funcs += 1
        return prefix

    def add_peak(self, func='pvoigt', guess=True, y=None, x=None):
        """
        Wrapper for MultiPeakFit.add_func to add a peak functions. 
        Default function is pvoigt. Model and Parameter will be 
        stored in the class instance.
        
        Parameters
        ----------
        func : string
            Function name
        guess : bool
            Guess initial parameter values
        y : array
            y data
        x : array
            x data
        
        Returns
        -------
        prefix : string
            Prefix of the new model parameters. Can be useful for
            modifying parameters after creation.
        """
        prefix = '{}{:02}_'.format(self._peak_prefix, self._n_peak_funcs + 1)
        model, param = self.new_func(func, prefix, guess, y, x)
        
        print(model, param)
        print(self._model, self._param)

        if self._model is None:
            self._model = model
            self._param = param
        else:
            self._model += model
            self._param += param

        parnames = ['amplitude', 'center', 'sigma']

        for par in parnames:
            self.set_parameter(prefix + par, {'min': 0})

        if func == 'pearson7':
            print('****************')
            self.set_parameter(prefix + 'expon', {'min': 1})

        for par in self._param.keys():
            print(self._param[par])

        self._n_peak_funcs += 1
        return prefix

    def pre_filter(self, y, x, plot=False):
        """
        Detect the number of peaks present in the data. Two arrays 
        are generated, filtered with a 1D Gaussian filter with wide
        (sigma1, + offset) and narrow (sigma2) widths. Peaks in the            
        data are present where filtered2 > filtered1. Labeled array
        and number of labels (peaks) are returned.
        
        Parameters
        ----------
        y : array
            y data
        x : array
            x data
        plot : bool
            Plot the output data
        
        Returns
        -------
        f1 : array
            Data filtered by self._fsigma1 and + self._foffset
        f2 : array
            Data filtered bu self._fsigma2
        fltd : array
            Filtered data showing location of detected peaks
        lbls : array
            Labeled array (i.e. location of peaks and their indices
        nps : int
            Number of labeled regions (i.e. peaks)
        """
        self.f1 = gauss1d(y, self._fsigma1) + self._foffset
        self.f2 = gauss1d(y, self._fsigma2)

        self.fltd = np.where(self.f2 > self.f1, 1, 0) * y.max()
        self.lbls, self.nps = label(self.fltd)

        if plot:
            self.plot_filter(y, x, self.f1, self.f2, self.fltd)

    def find_peaks(self, y, x, func='pvoigt', plot=False, plot_filtered=False):
        """
        Function to automatically find peaks and guess paramters.
        Data are first filtered with MultiPeakFit.pre_filter to 
        detect location of peaks. Then each peak is sequentially 
        guessed and added to self._model and self._param.
        
        Parameters
        ----------
        y : array
            y data
        x : array
            x data
        func : string
            Function name
        plot : bool
            Plot the output data
        """
        try:
            self.pre_filter(y, x, plot=plot_filtered)
            if self.nps < 1:
                raise RuntimeError
            for i in range(self.nps):
                print('Adding peak #%d' % i)
                ty = y[np.where(self.lbls == (i + 1))]
                tx = x[np.where(self.lbls == (i + 1))]
                self.add_peak(func=func, guess=True, y=ty, x=tx)
            if plot:
                guess = self._model.eval(params=self._param, x=x)
                comp = self._model.eval_components(params=self._param, x=x)
                self.plot_find_peaks(y, x, self.f1, self.f2, self.fltd,
                                     guess, comp)
            return True
        except RuntimeError:
            return False

    def set_parameter(self, parameter, val_dict):
        """
        Set any values of a parameter.
        
        Parameters
        ----------
        parameter : string
            Name of a parameter
        val_dict : dictionary
            Values to be changed in the parameter. 
            
            Allowed keys
            ------------
            value : float
                First guess of parameter's value
            vary : bool
                Allow parameter value to vary or not
            min : float
                Lower bound of allowed parameter value
            max : float
                Upper bound of allowed parameter value
            expr : string
                Mathematical expression to derrive paramter from 
                another
        """
        if (not isinstance(val_dict, dict)) or (parameter not in self._param.keys()):
            raise RuntimeError
        for i in range(len(val_dict)):
            self._param[parameter].set(**val_dict)

    def do_fit(self, y, x, pre_filter=True, require_errors=True,
               store_full=True, report=False,
               show_correl=False, plot=False, plot_filtered=False):
        """ 
        Main fit function. Function will abort if fit does not
        converge. If errors are requested (self._req_errors) 
        function will abort if errors cannot be estimated. If error
        threshold is specified function will abort if any errors
        are greater than the threshold. Results are reset at the
        start of the function. Function will return True if fit
        passes all tests. self._param is updated at the end (from 
        first guesses) to best fit parameters. This allows more
        efficient fitting when processing bulk data.
        
        Parameters
        -----------
        y : array
            y data
        x : array
            x data
        store_full : bool
            Store full results. If false only model.out, coeffs, 
            errors and fit will be stored.
        report : bool
            Print fit report
        show_correl : bool
            Show function correlations in fit report (does nothing
            if report = False)
        
        Returns
        -------
        bool
            Truth value of fit success based on any tests.
        """
        if (self._model is None) or (self._param is None):
            raise RuntimeError(('No fit model prepared. Hint: use '
                                'MultiPeakFit.find_peaks or add_peak'))
        self.reset_results()

        try:
            if pre_filter:
                self.pre_filter(y, x, plot_filtered)
                if self.nps is not self._n_peak_funcs:
                    print('Aborting: %g peaks expected, %g peaks found'
                          % (self._n_peak_funcs, self.nps))
                    raise RuntimeError

            print('Fitting...')
            out = self._model.fit(y, self._param, x=x)

            print('Fit converged:   ', out.success)
            print('Errors estimated:', out.errorbars)

            if report:
                print(out.fit_report(show_correl=show_correl))
            if not out.success:
                print('Aborting: fit did not converge')
                raise RuntimeError('Aborting: fit did not converge')
            if require_errors and (not out.errorbars):
                print('Aborting: errors could not be estimated')
                raise RuntimeError

            keys = out.params.keys()
            coeffs = np.asarray([out.params[key].value for key in keys])
            stderr = np.asarray([out.params[key].stderr for key in keys])

            if require_errors:
                if any(stderr / coeffs > self._err_thresh):
                    print('Aborting: unreasonable errors (> %g %%)' %
                          (self._err_thresh * 100))
                    raise RuntimeError
        except RuntimeError:
            self.reset_results()
            return False
        
        self._param = out.params
        self.out = out
        self.coeffs = coeffs
        self.stderr = stderr
        self.fit = out.best_fit
        
        if store_full or plot:
            self.res = out.residual
            self.init = out.init_fit
            self.comp = out.eval_components()
            self.covar = out.covar
            self.store_peaks_bkg()

        if plot:
            self.plot_fit(y, x)
        return True

    def store_peaks_bkg(self):
        """
        Stores total background and individual background components
        in class instance in self.bkg (array) and self.bkg_comp (dict 
        of arrays). Looks through self.comp for prefixes matching 
        self._bkg_prefix to determine what is from background.
        """
        self.peaks = {}
        self.bkg = None
        self.bkg_comp = {}
        bkg = np.zeros(self.fit.shape)

        for key in self.comp.keys():
            if key.startswith(self._bkg_prefix):
                self.bkg_comp[key[:-1]] = self.comp[key]
                bkg += self.comp[key]
            else:
                self.peaks[key[:-1]] = self.comp[key]
        self.bkg = bkg
    
    def remove_peak(self, prefix):
        """ """
        model = self._model
        param = self._param
        
        if prefix not in [m.prefix for m in model.components]:
            raise RuntimeError('{} not in peak prefixes'.format(peak))
        
        model_list = [c for c in model.components if c.prefix != prefix]
               
        tmp_model = model_list[0]
        for i in range(1, len(model_list)):
            tmp_model += model_list[i]
            
        pidx = 1
        changed_names = {}
        for c in tmp_model.components:
            if c.prefix.startswith('peak'):
                new_prefix = 'peak{:02}_'.format(pidx)
                changed_names[new_prefix] = c.prefix
                c.prefix = new_prefix
                pidx += 1
        
        tmp_param = tmp_model.make_params()
        for p in tmp_param:
            if p in changed_names.keys():
                prefix, par = p.split('_')
                par_name = '{}_{}'.format(changed_names[prefix], par)
            else:
                par_name = p
            tmp_param[p] = param[par_name]
        
        self._model = tmp_model
        self._param = tmp_param
        self._n_bkg_funcs = len([p for p in self._model.components \
                                     if p.prefix.startswith('bkg')])
        self._n_peak_funcs = len([p for p in self._model.components \
                                      if p.prefix.startswith('peak')])
    
    
    # --------------------------------------------------------------------------
    #     Functions for loading and saving models
    # --------------------------------------------------------------------------
    
    def load_model(self, model_file, x=None, y=None):
        """ 
        Load a pre-made model from a file. Models can be stored as
        '.yaml' files (requires the yaml library to read or '.json'.
        If the 'guess' parameter for a function is True (only valid 
        for yaml files), x and y data must be passed to estimate 
        parameters. yaml file can specify minimal data (as first guess),
        json files are complete dump of previous model.
        
        args:
            model_file (string): path to file containing model.
            x (ndarray, optional: x data, required only if guessing parameters.
            y (ndarray, optional: y data, required only if guessing parameters.
        """
        self.reset()
        if model_file.endswith('json'):
            raise NotImplementedError('json loading not implemented yet.')
        elif model_file.endswith('yaml'):
            if not has_yaml:
                raise RuntimeError('yaml library was not imported.')
            self.load_yaml(model_file, x, y)
        else:
            raise RuntimeError('Unsupported file type')
        return 1
    
    def save_model(self, filename):
        """ """
        raise NotImplementedError('Model saving not implemented')
        extension = os.path.splitext(filename)[1]
        if extension == '.yaml':
            self.save_yaml(filename)
        elif extension == '.json':
            self.save_json(filename)
        else:
            raise RuntimeError('{} is not a valid file format, must use'
                               '.yaml or .json')
        return 1
    
    def load_yaml(self, model_file, x, y):
        """ """
        with open(model_file, 'r') as f:
            tmp = yaml.load(f)
        
        background = tmp['background']
        peaks = tmp['peaks']
        
        self.load_yaml_params(background, 'bkg', x, y)
        self.load_yaml_params(peaks, 'peak', x, y)
        return 1
    
    def load_yaml_params(self, dictionary, mode, x, y):
        """ """
        for prefix in sorted(dictionary.keys()):
            function = dictionary[prefix]
            func = function['func']
            
            if 'guess' in function.keys() and function['guess']:
                guess = True
            else:
                guess = False
            
            tmp_model, tmp_param = self.new_func(func, prefix+'_', 
                                                 guess=guess, x=x, y=y)
            
            if self._model is None:
                self._model = tmp_model
                self._param = tmp_param
            else:
                self._model += tmp_model
                self._param += tmp_param
            
            if 'parameters' in function.keys():
                param_dict = function['parameters']
                for param_key in param_dict.keys():
                    parameter_name = '{}_{}'.format(prefix, param_key)
                    val_dict = self.format_val_dict(param_dict[param_key])
                    self.set_parameter(parameter_name, val_dict)
       
            if mode == 'bkg':
                self._n_bkg_funcs += 1
            elif mode == 'peak':
                self._n_peak_funcs += 1
            else:
                self.reset()
                raise RuntimeError('Unrecognised mode: {}'.format(mode))
        return 1            
    
    def format_val_dict(self, val_dict):
        """ """
        for val_key in val_dict.keys():
            val = val_dict[val_key]
            if (val.__class__ is str) and ('inf' in val):
                if val[0] == '-':
                    val_dict[val_key] = -np.inf
                else:
                    val_dict[val_key] = np.inf
        return val_dict
    
    # --------------------------------------------------------------------------
    #     Plotting functions
    # --------------------------------------------------------------------------

    def plot_filter(self, y, x, f1, f2, filtered, legend=False,
                    xlabel=None, ylabel=None,
                    lw=1.5, ms=6, fa=0.5,
                    filename=None):
        """
        """
        fig1 = plt.figure(1, figsize=(7, 6))
        self.cmd_plot_filtered(fig1, y, x, f1, f2, filtered, legend,
                               xlabel, ylabel, lw, ms, fa)
        if filename:
            print("Save figure not implemented")
        plt.show()

    def plot_guess(self, y, x, guess, comp, legend=False,
                   xlabel=None, ylabel=None,
                   lw=1.5, ms=6, fa=0.5,
                   filename=None):
        """
        """
        fig1 = plt.figure(1, figsize=(7, 6))
        self.cmd_plot_guess(fig1, y, x, guess, comp, legend,
                            xlabel, ylabel, lw, ms, fa)
        if filename:
            print("Save figure not implemented")
        plt.show()

    def plot_find_peaks(self, y, x, f1, f2, filtered, guess, comp,
                        legend=False, xlabel=None, ylabel=None,
                        lw=1.5, ms=6, fa=0.5,
                        filename=None):
        """
        """
        fig1 = plt.figure(1, figsize=(14, 6))

        plt.subplot(121)
        self.cmd_plot_filtered(fig1, y, x, f1, f2, filtered, legend,
                               xlabel, ylabel, lw, ms, fa)

        plt.subplot(122)
        self.cmd_plot_guess(fig1, y, x, guess, comp, legend,
                            xlabel, ylabel, lw, ms, fa)

        if filename:
            print("Save figure not implemented")
        plt.show()

    def plot_fit(self, y, x, fit=True,
                 res=True, bkg=True,
                 bkg_comp=False, init=False,
                 legend=True, xlim=None, ylim=None,
                 xlabel=None, ylabel=None, rlabel=None,
                 lw=1.5, ms=6, fa=0.5,
                 filename=None):
        """
        Plot the results of a PeakFit.peak_fit.
        
        Parameters
        ----------
        y, x : arrays
            Input x and y data
        fit : array
            Fitted curve
        res : array
            Residuals
        bkg = background (optional)
        """
        fig1 = plt.figure(1, figsize=(9, 7))
        frame1 = fig1.add_axes((.1, .3, .8, .6))

        # Data plot
        ldata, = plt.plot(x, y, 'ko', markersize=ms, label='data')
        handles = [ldata]

        if fit:
            if self.fit is None:
                raise RuntimeError(('MultiPeakFit.plot_fit: fit data not '
                                    'found!'))
            else:
                lfit, = plt.plot(x, self.fit, 'r-', markersize=ms, linewidth=lw,
                                 label='Fit')
                handles.append(lfit)
        if bkg:
            if self.bkg is None:
                print('MultiPeakFit.plot_fit: background data not found!')
            else:
                # for key in self.bkg.keys():
                lbkg, = plt.plot(x, self.bkg, 'g', linewidth=lw,
                                 label='Background')
                handles.append(lbkg)
        if self.peaks:
            if self.peaks is None:
                print('MultiPeakFit.plot_fit: peak data not found!')
            else:
                for key in self.peaks.keys():
                    lpeak, = plt.plot(x, self.peaks[key], 'b', linewidth=lw,
                                      label='Peaks')
                handles.append(lpeak)
        if bkg_comp:
            if self.bkg_comp is None:
                print('MultiPeakFit.plot_fit: background component data not'
                      ' found!')
            else:
                for key in self.bkg_comp.keys():
                    lbcom, = plt.plot(x, self.bkg_comp[key], 'g-', linewidth=lw,
                                      label='Background')
                handles.append(lbcom)
        if init:
            if self.init is None:
                print('MultiPeakFit.plot_fit: initial guess data not found!')
            else:
                lini, = plt.plot(x, self.init, 'b-', linewidth=lw,
                                 label='Initial guess')
                handles.append(lini)

        if ylabel:
            plt.ylabel(ylabel)
        frame1.set_xticklabels([])
        plt.grid()

        if legend:
            plt.legend(handles=handles, loc=2)

        # Residual plot
        if res:
            if self.res is None:
                print('MultiPeakFit.plot_fit: residual data not found!')
            else:
                frame2 = fig1.add_axes((.1, .1, .8, .2))
                lres, = plt.plot(x, self.res, 'or', markersize=ms,
                                 label='Residuals')
                handles.append(lres)
                plt.fill_between(x, self.res, where=self.res >= 0,
                                 facecolor='red', alpha=fa, interpolate=True)
                plt.fill_between(x, self.res, where=self.res <= 0,
                                 facecolor='red', alpha=fa, interpolate=True)

        if rlabel:
            plt.ylabel(rlabel)
        if xlabel:
            plt.xlabel(xlabel)
        plt.grid()
        plt.show()

    def cmd_plot_filtered(self, fig, y, x, f1, f2, filtered, legend,
                          xlabel, ylabel, lw, ms, fa):
        """
        """
        label_f1 = 'sigma = %g, offset = %g' % (self._fsigma1, self._foffset)
        label_f2 = 'sigma = %g' % (self._fsigma2)

        plt.plot(x, y, 'ko', markersize=ms, label='data')
        plt.plot(x, f1, linewidth=lw, label=label_f1)
        plt.plot(x, f2, linewidth=lw, label=label_f2)
        plt.plot(x, filtered, linewidth=lw, label='filtered')
        if legend:
            plt.legend(loc=2)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
    
    def cmd_plot_guess(self, fig, y, x, guess, comp, legend,
                       xlabel, ylabel, lw, ms, fa):
        """
        """
        ldata, = plt.plot(x, y, 'ko', markersize=ms, label='data')
        lguess, = plt.plot(x, guess, 'r', linewidth=lw, label='first guess')

        for key in comp.keys():
            if key.startswith(self._bkg_prefix):
                lbkg, = plt.plot(x, comp[key], 'g', linewidth=lw,
                                 label=self._bkg_prefix)
            else:
                lpeak, = plt.plot(x, comp[key], 'b', linewidth=lw,
                                  label=self._peak_prefix)
        if legend:
            plt.legend(loc=1, handles=[ldata, lguess, lbkg, lpeak])
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
