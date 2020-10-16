#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QUOCKA_SIMULATE

Tools for simulating QUOCKA data
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

C = 299792458.  # m/s


class RealSource:
    """Store data from a real QUOCKA source.
    """    

    def __init__(self):
        self.data = {}
        self.i_fit = None

    def read_data(self, filename, reformatted=False):
        """Read data from QUOCKA source file.

        Args:
            filename (str): Name of source
            reformatted (bool, optional): Source file is reformatted. 
                Defaults to False.
        """
        values = np.loadtxt(filename)
        ind = np.argsort(values[:, 0])
        if reformatted:
            self.data['freq'] = values[ind, 0]
            self.data['lamsq'] = (C/values[ind, 0])**2
            self.data['I'] = values[ind, 1]
            self.data['Q'] = values[ind, 2]
            self.data['U'] = values[ind, 3]
            self.data['Ierr'] = values[ind, 4]
            self.data['Qerr'] = values[ind, 5]
            self.data['Uerr'] = values[ind, 6]
        else:
            self.data['freq'] = values[ind, 0]*1.e9
            self.data['lamsq'] = (C/(values[ind, 0]*1.e9))**2
            self.data['I'] = values[ind, 1]
            self.data['Ierr'] = values[ind, 2]
            self.data['Q'] = values[ind, 3]
            self.data['Qerr'] = values[ind, 4]
            self.data['U'] = values[ind, 5]
            self.data['Uerr'] = values[ind, 6]
            self.data['V'] = values[ind, 7]
            self.data['Verr'] = values[ind, 8]

    def freq_subset(self, fmin=None, fmax=None, inplace=False):
        """Get a subset of the data, using a frequency cut.

        Args:
            fmin (float, optional): Minimum frequency of cut. Defaults to None.
            fmax (float, optional): Maximum frequency of cut. Defaults to None.
            inplace (bool, optional): Update data in place. Defaults to False.

        Returns:
            RealSource: Cut down RealSource object.
        """
        if fmin is None:
            fmin = np.amin(self.data['freq'])
        if fmax is None:
            fmax = np.amax(self.data['freq'])
        ind = np.where(np.logical_and(
            self.data['freq'] >= fmin, self.data['freq'] <= fmax))
        if inplace:
            for k in list(self.data.keys()):
                self.data[k] = self.data[k][ind]
        else:
            subset = self.copy()
            for k in list(subset.data.keys()):
                subset.data[k] = subset.data[k][ind]
            return subset

    def fit_imodel(self, order=3, show_plot=True):
        """Fit polynomial to Stokes I spectrum.

        Args:
            order (int, optional): Order of fit for polyval. Defaults to 3.
            show_plot (bool, optional): Show plot of model fit. Defaults to True.

        Returns:
            list: Polyfit values.
        """        
        i_fit = np.polyfit(np.log10(self.data['freq']),
                           np.log10(self.data['I']),
                           order
                           )
        if show_plot:
            plt.figure(facecolor='w')
            plt.plot(self.data['freq'],
                     self.data['I'],
                     '.',
                     label='Data'
                     )
            plt.plot(self.data['freq'],
                     10.**np.polyval(i_fit, np.log10(self.data['freq'])),
                     'r-',
                     label='Model'
                     )
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Stokes I [Jy/beam]')
            plt.legend()

        self.i_fit = i_fit
        return i_fit


class SimulatedSource:
    """Generate a simulated QUOCKA source
    """    
    def __init__(self,
                 template=None,
                 freq=None,
                 inoise=None,
                 inoisestd=None,
                 qnoise=None,
                 qnoisestd=None,
                 unoise=None,
                 unoisestd=None):
        """SimulatedSource

        Args:
            template (RealSource, optional): Source to use as template. Defaults to None.
            freq (array-like, optional): List of frequencies. Defaults to None.
            inoise (float, optional): Mean noise in Stokes I. Defaults to None.
            inoisestd (float, optional): Std of noise in Stokes I. Defaults to None.
            qnoise (float, optional): Mean noise in Stokes Q. Defaults to None.
            qnoisestd (float, optional): Std of noise in Stokes Q. Defaults to None.
            unoise (float, optional): Mean noise in Stokes Q. Defaults to None.
            unoisestd (float, optional): Std of noise in Stokes Q. Defaults to None.

        Raises:
            ValueError: If no template or frequency is provided.
            ValueError: If a template and frequency are provided.
        """                
        self.data = {}
        self.model = {}
        self.model['I'] = {}
        self.model['QU'] = {}
        self.nmodels = [0, 0]
        self.model_fdf = {}
        self.model_fdf['phi'] = []
        self.model_fdf['data'] = []
        if template is None and freq is None:
            print('You must either provide a template or specify the frequencies to use')
            raise ValueError
        elif template is not None and freq is not None:
            print('You must provide either a template or frequencies to use, not both')
            raise ValueError
        elif freq is None:
            assert type(template).__module__ == type(
                self).__module__, "Template must be a quocka_simulate object"
            self.data['freq'] = template.data['freq']
            self.data['lamsq'] = template.data['lamsq']
            if not [inoise, qnoise, unoise] == [None, None, None]:
                self.noise = {
                    'I': inoise,
                    'Q': qnoise,
                    'U': unoise
                }
            else:
                self.noise = {
                    'I': np.nanmedian(template.data['Ierr']),
                    'Q': np.nanmedian(template.data['Qerr']),
                    'U': np.nanmedian(template.data['Uerr'])
                }
            if not [inoise, qnoise, unoise] == [None, None, None]:
                self.noisestd = {
                    'I': inoisestd,
                    'Q': qnoisestd,
                    'U': unoisestd
                }
            else:
                self.noisestd = {
                    'I': np.nanstd(template.data['Ierr']),
                    'Q': np.nanstd(template.data['Qerr']),
                    'U': np.nanstd(template.data['Uerr'])
                }
        else:
            assert type(
                freq).__module__ == np.__name__, "Frequencies must be provided as a numpy array"
            assert len(freq.shape) == 1, "Frequency array must be 1-D"
            self.data['freq'] = np.sort(freq)
            self.data['lamsq'] = (C/self.data['freq'])**2
            self.noise = {'I': inoise, 'Q': qnoise, 'U': unoise}
            self.noisestd = {'I': inoisestd, 'Q': qnoisestd, 'U': unoisestd}

        self.data['I'] = np.zeros(self.data['freq'].shape)
        self.data['Q'] = np.zeros(self.data['freq'].shape)
        self.data['U'] = np.zeros(self.data['freq'].shape)

        # TODO: add a way to create and plot the FDF

    def __add__(self, y):
        """[summary]

        Args:
            y ([type]): [description]

        Raises:
            NotImplementedError: [description]
            NotImplementedError: [description]

        Returns:
            [type]: [description]
        """        
        # TODO: This needs a way to combine models (not just data)
        # TODO: Also deal with noise and noisestd somehow!
        lm = len(self.data['freq'] == y.data['freq'])
        result = self.copy()
        if lm == 0:
            print("Append mode")
            for k in list(self.data.keys()):
                result.data[k] = np.append(self.data[k], y.data[k])
            ind = np.argsort(result.data['freq'])
            for k in list(result.data.keys()):
                result.data[k] = result.data[k][ind]
        elif lm == len(self.data['freq']):
            print("Add mode")
            # check frequencies
            if np.sum(np.abs(self.data['freq']-y.data['freq'])) > 0.:
                print("Frequencies must be the same if they overlap")
                raise NotImplementedError
            for k in list(self.data.keys()):
                if k == 'freq':
                    pass
                else:
                    result.data[k] = self.data[k] + y.data[k]
        else:
            print("Frequencies must either overlap or be different")
            raise NotImplementedError
        return result

    def set_noise(self, **kwargs):
        """[summary]
        """        
        for key, value in list(kwargs.items()):
            if key == 'inoise':
                self.noise['I'] = value
            elif key == 'qnoise':
                self.noise['Q'] = value
            elif key == 'unoise':
                self.noise['U'] = value
            elif key == 'inoisestd':
                self.noisestd['I'] = value
            elif key == 'qnoisestd':
                self.noisestd['Q'] = value
            elif key == 'unoisestd':
                self.noisestd['U'] = value
            else:
                print(('WARNING: key', key, 'ignored'))

    def write_data(self, filename, reformatted=False):
        """Write simlated data to disk

        Args:
            filename (str): Output file name.
            reformatted (bool, optional): Use formatted format. Defaults to False.
        """        
        if reformatted:
            datapack = np.zeros((len(self.data['freq']), 7))
            datapack[:, 0] = self.data['freq']
            datapack[:, 1] = self.data['Iobs']
            datapack[:, 2] = self.data['Qobs']
            datapack[:, 3] = self.data['Uobs']
            datapack[:, 4] = self.data['Ierr']
            datapack[:, 5] = self.data['Qerr']
            datapack[:, 6] = self.data['Uerr']
        else:
            datapack = np.zeros((len(self.data['freq']), 9))
            datapack[:, 0] = self.data['freq']/1.e9
            datapack[:, 1] = self.data['Iobs']
            datapack[:, 2] = self.data['Ierr']
            datapack[:, 3] = self.data['Qobs']
            datapack[:, 4] = self.data['Qerr']
            datapack[:, 5] = self.data['Uobs']
            datapack[:, 6] = self.data['Uerr']
        np.savetxt(filename, datapack)

    def add_stokesi(self, pvals, log=True):
        """Create Stokes I data.

        Args:
            pvals (list): Values for polyfit.
            log (bool, optional): Create spectrum in log-space. Defaults to True.
        """        
        if log:
            values = np.polyval(pvals, np.log10(self.data['freq']))
            nl = 10.**values
        else:
            nl = np.polyval(pvals, self.data['freq'])
        self.data['I'] += nl
        self.model['I']['I%d' % (self.nmodels[0])] = [pvals, log]
        self.nmodels[0] += 1

    def add_simple_rm(self, pfrac, rm, chi0, verbose=False):
        """Add a simple RM model.

        Tip - use values from RealSource.fit_imodel

        Args:
            pfrac (float): Polarisation fraction.
            rm (float): Rotation Measure.
            chi0 (float): Polarisation angle.
            verbose (bool, optional): Print out info. Defaults to False.
        """        
        if verbose:
            print(('Polarization fraction is', pfrac))
            print(('Intrinsic pol angle is', chi0, 'deg'))
        chi0 *= np.pi/180.
        if verbose:
            print(('Intrinsic pol angle is', chi0, 'rad'))
            print(('RM is', rm, 'rad/m2'))
        pvals = pfrac*self.data['I']*np.exp(2.j*(chi0+rm*self.data['lamsq']))
        self.data['Q'] += np.real(pvals)
        self.data['U'] += np.imag(pvals)
        self.model['QU']['QU%d' % (self.nmodels[1])] = [
            'simple', pfrac, rm, chi0]
        self.nmodels[1] += 1

    def add_dfr(self, pfrac, R, rm, chi0):
        """Add differential Faraday rotation model (Burn slab).

        Args:
            pfrac (float): polarisation fraction.
            R (float): Faraday dpeth.
            rm (float): Rotation Measure.
            chi0 (float): Intrinsic polarisation angle.
        """        
        print(('Polarization fraction is', pfrac))
        print(('Intrinsic pol angle is', chi0, 'deg'))
        chi0 *= np.pi/180.
        print(('Intrinsic pol angle is', chi0, 'rad'))
        print(('Faraday depth is', R, 'rad/m2'))
        # print('Additional RM is',rm,'rad/m2')
        print(('Effective RM is', rm, 'rad/m2'))
        # pvals = pfrac*self.data['I']*np.sin(R*self.data['lamsq'])/(R*self.data['lamsq'])*np.exp(2.j*(chi0+0.5*R*self.data['lamsq']+rm*self.data['lamsq']))
        pvals = pfrac*self.data['I']*np.sin(R*self.data['lamsq'])/(
            R*self.data['lamsq'])*np.exp(2.j*(chi0+rm*self.data['lamsq']))
        self.data['Q'] += np.real(pvals)
        self.data['U'] += np.imag(pvals)
        self.model['QU']['QU%d' % (self.nmodels[1])] = [
            'dfr', pfrac, R, rm, chi0]
        self.nmodels[1] += 1

    def add_ext(self, pfrac, sig, rm, chi0):
        """Add external Faraday dispersion model

        Args:
            pfrac (float): Polarisation fraction.
            sig (float): Faraday dispersion.
            rm (float): Rotation Measure.
            chi0 (float): Intrinstic polarisation angle.
        """
        print(('Polarization fraction is', pfrac))
        print(('Intrinsic pol angle is', chi0, 'deg'))
        chi0 *= np.pi/180.
        print(('Intrinsic pol angle is', chi0, 'rad'))
        print(('Dispersion in rm is', sig, 'rad/m2'))
        # print('Additional RM is',rm,'rad/m2')
        print(('Effective RM is', rm, 'rad/m2'))
        pvals = pfrac*self.data['I']*np.exp(-2.*sig**2*self.data['lamsq']**2)*np.exp(
            2.j*(chi0+rm*self.data['lamsq']))
        self.data['Q'] += np.real(pvals)
        self.data['U'] += np.imag(pvals)
        self.model['QU']['QU%d' % (self.nmodels[1])] = [
            'ext', pfrac, sig, rm, chi0]
        self.nmodels[1] += 1

    def add_mix(self, pfrac, R, sig, rm, chi0):
        """Add a mixture model.

        Args:
            pfrac (float): Polarisation fraction
            R (float): Faraday depth.
            sig (float): Faraday dispersion.
            rm (float): Rotation Measure.
            chi0 (float): Intrinsic polarisation anlge.
        """        
        print(('Polarization fraction is', pfrac))
        print(('Intrinsic pol angle is', chi0, 'deg'))
        chi0 *= np.pi/180.
        print(('Intrinsic pol angle is', chi0, 'rad'))
        print(('Faraday depth is', R, 'rad/m2'))
        print(('Dispersion in rm is', sig, 'rad/m2'))
        # print('Additional RM is',rm,'rad/m2')
        print(('Effective RM is', rm, 'rad/m2'))
        pvals = pfrac*self.data['I']*np.sin(R*self.data['lamsq'])/(R*self.data['lamsq']) * \
            np.exp(-2.*sig**2*self.data['lamsq']**2) * \
            np.exp(2.j*(chi0+rm*self.data['lamsq']))
        self.data['Q'] += np.real(pvals)
        self.data['U'] += np.imag(pvals)
        self.model['QU']['QU%d' % (self.nmodels[1])] = [
            'mix', pfrac, R, sig, rm, chi0]
        self.nmodels[1] += 1

    # def add_ifd(self, pfrac, R, rm, chi0, srm):
    #     print('Polarization fraction is',pfrac)
    #     print('Intrinsic pol angle is',chi0,'deg')
    #     chi0 *= np.pi/180.
    #     print('Intrinsic pol angle is',chi0,'rad')
    #     print('Faraday depth is',R,'rad/m2')
    #     print('Additional RM is',rm,'rad/m2')
    #     print('Internal Faraday dispersion is',srm,'rad/m2')
    #     pvals = pfrac*self.data['I']*np.exp(2.j*(chi0+rm*self.data['lamsq']))*((1.-np.exp(2.j*R*self.data['lamsq']-2.*srm**2*self.data['lamsq']**2))/(2.*srm**2*self.data['lamsq']**2-2.j*R*self.data['lamsq']))
    #     self.data['Q'] += np.real(pvals)
    #     self.data['U'] += np.imag(pvals)
    #     self.model['QU']['QU%d'%(self.nmodels[1])] = ['ifd', pfrac, R, rm, chi0, srm]
    #     self.nmodels[1] += 1

    def generate_model_fdf(self, phi):
        """Create model FDF

        Args:
            phi (array-like): Array of Faraday depths.
        """        
        model_fdf = np.zeros(phi.shape, dtype=np.complex)
        stokesI = np.zeros(phi.shape)
        glsq = np.arange(-1000, 1000.0005, 0.001)
        # currently just deal with entire stokes I model
        # TODO: figure out how to partition this (make models objects?)
        """
        for i in range(self.nmodels[0]):
            imodel = self.model['I']['I%d'%i]
            if imodel[1]: # log==True
                values = np.polyval(imodel[0], np.log10(self.data['freq']))
                stokesI += 10.**values
            else:
                stokesI += np.polyval(imodel[0], self.data['freq'])
        """
        for i in range(self.nmodels[1]):
            qumodel = self.model['QU']['QU%d' % i]
            if qumodel[0] == 'simple':
                # make a simple RM spectrum at the nearest phi pixel
                # warn if out of range
                pfrac, rm, chi0 = qumodel[1:]
                if rm < np.min(phi) or rm > np.max(phi):
                    print('Warning: Selected RM out of range!')
                else:
                    ind = np.where(np.abs(phi-rm) == np.min(np.abs(phi-rm)))
                    model_fdf[ind] += (pfrac*np.exp(1.j*chi0))
            elif qumodel[0] == 'dfr':
                # make a box spectrum
                # warn if out of range
                pfrac, R, rm, chi0 = qumodel[1:]
                if rm > np.max(phi) or (rm+R) < np.min(phi):
                    print('Warning: Selected RM out of range!')
                else:
                    # ind = np.where(np.logical_and(phi<=(rm+R),phi>=rm))
                    ind = np.where(np.logical_and(
                        phi <= (rm+0.5*R), phi >= rm-0.5*R))
                    model_fdf[ind] += (pfrac*np.exp(1.j*chi0))
            elif qumodel[0] == 'ifd':
                # make the complicated IFD spectrum
                # warn if out of range
                pfrac, R, rm, chi0, srm = qumodel[1:]
                if rm-srm > np.max(phi) or (rm+R+srm) < np.min(phi):
                    print('Warning: Selected RM out of range!')
                else:
                    ind = np.where(np.logical_and(phi <= (rm+R), phi >= rm))
                    model_fdf[ind] += (pfrac*np.exp(1.j*chi0))
                    if rm > np.min(phi):
                        ind = np.where(phi < rm)
                        amp = pfrac*np.exp(-(phi[ind]-rm)**2/(2.*srm**2))
                        model_fdf[ind] += (amp*np.exp(1.j*chi0))
                    if np.max(phi) > (rm+R):
                        ind = np.where(phi > (rm+R))
                        amp = pfrac*np.exp(-(phi[ind]-(rm+R))**2/(2.*srm**2))
                        model_fdf[ind] += (amp*np.exp(1.j*chi0))
            else:
                print(('Warning: not sure what this QU model is:', qumodel[0]))
                print('No model FDF generated.')
        self.model_fdf['phi'] = phi
        self.model_fdf['data'] = model_fdf

    def plot_model_fdf(self, pltfile=None):
        """Plot model FDF.

        Args:
            pltfile (str, optional): Outfile to save plot. Defaults to None.

        Raises:
            RuntimeError: If FDF model has not been generated.
        """        
        # check if model FDF exists
        if len(self.model_fdf['phi']) == 0:
            print('First you need to generate the model FDF with generate_model_fdf()')
            raise RuntimeError
        plt.figure(figsize=(10, 16), facecolor='w')
        plt.subplot(311)
        plt.plot(self.model_fdf['phi'], np.abs(self.model_fdf['data']), 'k-')
        plt.xlabel('RM')
        plt.ylabel('abs(FDF)')
        plt.subplot(312)
        plt.plot(self.model_fdf['phi'], np.real(self.model_fdf['data']), 'r-')
        plt.xlabel('RM')
        plt.ylabel('real(FDF)')
        plt.subplot(313)
        plt.plot(self.model_fdf['phi'], np.imag(self.model_fdf['data']), 'b-')
        plt.xlabel('RM')
        plt.ylabel('imag(FDF)')
        if pltfile is not None:
            plt.savefig(pltfile, bbox_inches='tight', dpi=200)

    def apply_noise(self):
        """Add noise to model I, Q, and U
        """        
        for stokes in ['I', 'Q', 'U']:
            self.data[stokes+'obs'] = self.data[stokes] + self.noise[stokes] * \
                np.random.standard_normal(self.data[stokes].shape)
            self.data[stokes+'err'] = self.noisestd[stokes] * \
                np.random.standard_normal(
                    self.data[stokes].shape) + self.noise[stokes]

    def plot_2x2(self, pltfile=None):
        """Plot noisy model I, Q and U, Q vs U, and PA

        Args:
            pltfile (str, optional): Outfile to save plot. Defaults to None.
        """        
        plt.figure(figsize=(14, 12), facecolor='w')
        plt.subplot(221)
        plt.errorbar(self.data['freq'], self.data['Iobs'],
                     yerr=self.data['Ierr'], marker='o', mfc='none', ls='none')
        plt.gca().set_yscale('log')
        plt.gca().set_xscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Stokes I')
        plt.subplot(222)
        plt.errorbar(self.data['lamsq'], self.data['Qobs'],
                     yerr=self.data['Qerr'], marker='o', mfc='none', ls='none')
        plt.errorbar(self.data['lamsq'], self.data['Uobs'],
                     yerr=self.data['Uerr'], marker='o', mfc='none', ls='none')
        plt.legend(('Q', 'U'), numpoints=1, loc='best')
        plt.xlabel('Wavelength squared')
        plt.ylabel('Stokes Q,U')
        plt.subplot(223)
        plt.scatter(self.data['Qobs'], self.data['Uobs'], marker='o',
                    c=self.data['lamsq'], s=32, cmap='rainbow', zorder=20)
        plt.errorbar(self.data['Qobs'], self.data['Uobs'], xerr=self.data['Qerr'],
                     yerr=self.data['Uerr'], c='k', ls='none', zorder=10)
        plt.axhline(ls='--', c='k', zorder=0)
        plt.axvline(ls='--', c='k', zorder=0)
        l, r = plt.xlim()
        b, t = plt.ylim()
        axm = np.max(np.abs(np.array([l, r, b, t])))
        plt.xlim(-axm, axm)
        plt.ylim(-axm, axm)
        plt.xlabel('Stokes Q')
        plt.ylabel('Stokes U')
        plt.subplot(224)
        plt.plot(self.data['lamsq'], 0.5*180./np.pi *
                 np.arctan2(self.data['Uobs'], self.data['Qobs']), 'ko')
        plt.xlabel('Wavelength squared')
        plt.ylabel('Pol angle')
        if pltfile is not None:
            plt.savefig(pltfile, bbox_inches='tight', dpi=200)
