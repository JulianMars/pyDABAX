# -*- coding: utf-8 -*-
"""
High level functions for energy and momentum dependent x-ray and neutron quantities.
"""

import logging
import numpy as np
from numpy import polyfit, poly1d, argmax, arange
import astropy.constants as c
import astropy.units as u
from astropy.units import Quantity
from pandas import DataFrame
import regex as re
from .dabax import dabax
import warnings

__author__ = "Julian Mars"
__copyright__ = "Julian Mars"
__license__ = "mit"

_logger = logging.getLogger(__name__)

hc = (c.h * c.c).to(u.kiloelectronvolt * u.Angstrom)
r_e = Quantity(c.si.e ** 2 / (4 * np.pi * c.eps0 * c.m_e * c.c ** 2), 'Å')


def show_unit(function):
    def wrapper(*args, **kwargs):
        func = function(*args, **kwargs)
        return func if UnitSettings.SHOW_UNIT else func.value

    return wrapper


def enforce_unit_iter(unit):
    def decorator(function):
        def wrapper(*args, **kwargs):
            if len(args) > 1:
                args = list(args)
                args[1] = unit_iter_parser(args[1], unit())

            result = function(*args, **kwargs)
            if result is not None:
                return Quantity(result, unit()).squeeze() if UnitSettings.SQUEEZE else Quantity(result, unit())

        return wrapper

    return decorator


def unit_iter_parser(inp, unit):
    try:
        res = Quantity(inp, unit)
    except:
        res = []
        for r in inp:
            res.append(Quantity(r, unit))
        res = Quantity(res, unit)

    try:
        iter(res)
    except:
        res = Quantity([res], unit)
    return res.flatten()


class UnitSettings:
    SHOW_UNIT = True
    SQUEEZE = True
    UNIT_Q = "Å^-1"
    UNIT_E = "keV"
    UNIT_TTH = "°"
    UNIT_R = "Å"
    UNIT_XS = "barn"
    UNIT_SL = "fm"


class BaseElement:

    def __init__(self, q="0 1/Å", photon_energy="8.047 keV"):
        self._q = None
        self._energy = None
        self._l = None
        self.energy = photon_energy
        self.q = q

    @property
    @show_unit
    @enforce_unit_iter((lambda: UnitSettings.UNIT_Q))
    def q(self):
        ans = self._q
        return ans

    @q.setter
    @enforce_unit_iter((lambda: UnitSettings.UNIT_Q))
    def q(self, value):
        self._q = value

    @property
    @show_unit
    @enforce_unit_iter((lambda: UnitSettings.UNIT_TTH))
    def ttheta(self):
        return self._ttheta()

    def _ttheta(self):
        ans = Element.calc_ttheta(self._energy, self._q)
        return ans

    @ttheta.setter
    @enforce_unit_iter((lambda: UnitSettings.UNIT_TTH))
    def ttheta(self, value):
        self._q = Element.calc_q(self.energy, value)

    @property
    @show_unit
    @enforce_unit_iter((lambda: UnitSettings.UNIT_E))
    def energy(self):
        ans = self._energy
        return ans

    @energy.setter
    @enforce_unit_iter((lambda: UnitSettings.UNIT_E))
    def energy(self, value):
        self._energy = value

    @staticmethod
    def calc_dabax(q, df, d=5):
        def f0():
            a, b, const = params
            summ = 0
            for _a, _b in zip(a, b):
                summ += _a * np.exp(-_b * k ** 2)
            summ += const
            return summ

        params = [df.values[0, :d],  # a
                  df.values[0, d + 1:],  # b
                  df.values[0, d]]  # c

        k = Quantity(q / (4 * np.pi), "1/Å").value  # Do not change; TABLES IN Å-1
        res = f0()
        return res

    @staticmethod
    def interpolate_chantler(energy, df):
        nist = df.values.astype(float).T
        arr = np.abs(nist[0] - energy)
        idx = arr.argsort()[:2]  # find nearest data-points
        out = []

        def interpolate():
            pass

        i = 1

        p1 = poly1d(
            polyfit(
                nist[0, idx],
                nist[i, idx],
                1,
            )
        )
        out.append(p1(energy))

        i = 2
        """ overflow at absorption edge, no np.float128 in windows?
        p2 = polyfit(  # nist recommends log-log for f2
            np.log10(nist[0, idx]),
            np.log10(nist[i, idx]),
            1,
        )
        print(nist[0, idx], nist[i, idx])
        print((Energy.dtype, p2[0].dtype, p2[1].dtype))
        res = (lambda E, m, b: E ** m * 10 ** b)(float(Energy), float(p2[0]), float(p2[1]))

        out.append(res)
        """

        p1 = poly1d(
            polyfit(
                nist[0, idx],
                nist[i, idx],
                1,
            )
        )
        out.append(p1(energy))

        return out

    @staticmethod
    def calc_q(energy, ttheta):
        energy = Quantity(energy, UnitSettings.UNIT_E)
        wavelength = Quantity(hc / energy, UnitSettings.UNIT_R)
        ttheta = Quantity(ttheta, UnitSettings.UNIT_TTH)
        q = 4 * np.pi / wavelength * np.sin(ttheta / 2)
        return q

    @staticmethod
    def calc_ttheta(energy, q):
        energy = Quantity(energy, UnitSettings.UNIT_E)
        wavelength = Quantity(hc / energy, UnitSettings.UNIT_R)
        q = Quantity(q, UnitSettings.UNIT_Q)
        wavelength = Quantity(wavelength, UnitSettings.UNIT_R)
        try:
            x = np.tile(q, (len(wavelength), 1))
            y = np.tile(wavelength, (len(q), 1)).T
        except TypeError:
            x = q
            y = wavelength

        res = 2 * np.arcsin(x * y / (4 * np.pi))
        return Quantity(res, UnitSettings.UNIT_TTH)

    @staticmethod
    def crossec_compton_kleinnishina(energy, ttheta):
        energy = Quantity(energy, UnitSettings.UNIT_E)
        ttheta = Quantity(ttheta, UnitSettings.UNIT_TTH)
        d = 1 + (energy / (c.m_e * c.c ** 2)) * (
                1 - np.cos(ttheta))
        p = 1 / d  # P Klein-Nishina Equation
        diff_crossec = 1 / 2 * p ** 2 * (p + 1 / p - np.sin(ttheta) ** 2)
        return diff_crossec, energy * p

    @property
    @show_unit
    @enforce_unit_iter((lambda: UnitSettings.UNIT_R))
    def wavelength(self):
        ans = hc / self._energy
        return ans  # if Element.SHOW_UNIT else ans.value

    @wavelength.setter
    @enforce_unit_iter((lambda: UnitSettings.UNIT_R))
    def wavelength(self, value):
        self._energy = Quantity(hc / value, UnitSettings.UNIT_E)


class Element(BaseElement):
    def __init__(self, symbol, q="0/Å", energy="8.047 keV"):
        super().__init__(q, energy)
        self.symbol = symbol
        self.element_symbol = dabax.get_entry(self.symbol, "element_symbol")

    def __repr__(self):
        return str(self.symbol) + '\n' + "Atomic weight: %s"%self.atomic_weight + '\n' + "X-ray formfactor: %f"%self.f.flat[0].real

    def _repr_html_(self):

        try:
            ed = self.edges._repr_html_()
        except KeyError:
            ed = ''

        keys = [["Symbol", self.symbol],
                ["Atomic number", self.atomic_number],
                ["Atomic mass", self.atomic_weight],
                ["Charge", self.charge],
                ["Atomic radius", self.atomic_radius],
                ["Covalent radius", self.covalent_radius],
                ["Melting point", self.melting_point],
                ["Boiling point", self.boiling_point],

                ["Energy", self._energy[0]],
                ["q", self._q[0]],
                ["X-ray formfactor", '-' if self.f.flat[0] == '-' else self.f.flat[0].real * u.electron],
                ["K<sub>α1</sub>", self.k_alpha_1],
                ["K<sub>α2</sub>", self.k_alpha_2],
                ["K<sub>β</sub>", self.k_beta],
                ["b<sub>coh</sub>", self.neutron_b_coh],
                ["b<sub>inc</sub>", self.neutron_b_inc],
                ["σ<sub>coh</sub>", self.neutron_xs_coh],
                ["σ<sub>inc</sub>", self.neutron_xs_inc],
                ["absorption (2200m/s)", self.neutron_abs_at_2200mps]]

        tab1 = ["<tr> <th>{}</th> <td>{:}</td> </tr>".format(name,
                                                             value if
                                                             isinstance(value, (str, int)) else value.round(4)) for
                name, value in keys]
        ans = ("<h1>{}</h1>".format(self.name) +
               "<table> " +
               "".join(tab1) +
               "</table>" + ed)
        return ans

    @property
    @show_unit
    def atomic_radius(self):
        try:
            val = self.atomic_constants['values'].loc['AtomicRadius (Å)']
        except KeyError:
            return '-'
        return Quantity(val, 'Å')

    @property
    @show_unit
    def covalent_radius(self):
        try:
            val = self.atomic_constants['values'].loc['CovalentRadius (Å)']
        except KeyError:
            return '-'
        return Quantity(val, 'Å')

    @property
    @show_unit
    def boiling_point(self):
        try:
            val = self.atomic_constants['values'].loc['BoilingPoint (K)']
        except KeyError:
            return '-'
        return Quantity(val, 'K')

    @property
    @show_unit
    def melting_point(self):
        try:
            val = self.atomic_constants['values'].loc['MeltingPoint (K)']
        except KeyError:
            return '-'
        return Quantity(val, 'K')

    @property
    @show_unit
    def density(self, database='auto'):
        dbs = {'neutron_booklet': self._density_neutron_booklet,
               'atomic_constants': self._density_atomic_constants}
        if database == 'auto':
            for k in dbs:
                ans = dbs[k]()
                if ans:
                    break
        else:
            db = dbs[database]
            ans = db()
        return ans if ans else '-'

    def _density_atomic_constants(self):
        try:
            val = self.atomic_constants['values'].loc['Density (g/ccm)']
        except KeyError:
            return None
        return Quantity(val, 'g/cm^3')

    def _density_neutron_booklet(self):
        try:
            val = self.neutron_booklet['SCLC Data Booklet'].loc['Density (g/cm3)']
        except KeyError:
            return None
        return Quantity(val, 'g/cm^3')

    @property
    @show_unit
    def debeye_temperature(self):
        val = self.atomic_constants['values'].loc['DebeyeTemperature (K)']
        return Quantity(val, 'K')

    @property
    @show_unit
    def thermal_conductivity(self):
        val = self.atomic_constants['values'].loc['ThermalConductivity (W/cmK)']
        return Quantity(val, 'W/cmK')

    @property
    @show_unit
    def atomic_volume(self):
        val = self.atomic_constants['values'].loc['AtomicVolume']
        return Quantity(val, 'cm^3/mol')

    @property
    def atomic_number(self):
        val = dabax.get_entry(self.symbol, "atomic_number")
        return val

    @property
    def mcgowan_volume(self):
        val = dabax.get_entry(self.symbol, "mcgowan_vol")
        return Quantity(val, 'Å^3')

    @property
    def z(self):
        return self.atomic_number

    @property
    def electron_number(self):
        return self.z - self.charge

    @property
    def charge(self):
        return dabax.get_entry(self.symbol, "charge")

    @property
    def mass_number(self):
        ans = dabax.get_entry(self.symbol, "mass_number")
        return ans

    @property
    def atomic_weight(self, database='auto'):
        dbs = {'neutron_booklet': self._atomic_weight_neutron_booklet, 'dabax': self._atomic_weight_dabax,
               'atomic_constants': self._atomic_weight_atomic_constants}
        if database == 'auto':
            for k in dbs:
                ans = dbs[k]()
                if ans:
                    break
        else:
            db = dbs[database]
            ans = db()
        return ans if ans else '-'

    def _atomic_weight_dabax(self):
        try:
            ans = dabax.get_entry(self.symbol, "atomic_weight")
        except KeyError:
            return None
        return Quantity(ans, u.misc.u)

    def _atomic_weight_atomic_constants(self):
        try:
            ans = self.atomic_constants['values'].loc['AtomicMass']
        except KeyError:
            return None
        return Quantity(ans, u.misc.u)

    def _atomic_weight_neutron_booklet(self):
        try:
            ans = self._get_dabax_SLCS_DataBooklet().loc['RelAtMass (g/mol)'][0]
        except KeyError:
            return None
        return (Quantity(ans, 'g/mol') / c.N_A).to(u.misc.u)

    @property
    def molecular_weight(self):
        return self.atomic_weight

    @property
    @show_unit
    def molar_mass(self):
        return (self.atomic_weight * c.N_A).to('g/mol')

    @property
    def f(self):
        try:
            ans = self.get_f(self._energy, self._q)
        except(KeyError):
            ans = ['-']
        return np.squeeze(ans) if UnitSettings.SQUEEZE else ans

    @property
    def crossec_compton(self):
        return self.get_compton(self._energy, self._q)

    @property
    def crossec_thomson(self):
        return self.get_thomson(self._energy, self._q)

    @property
    def edges(self):
        return self._get_nist_edges_chantler()

    @property
    def atomic_constants(self):
        return self._get_dabax_atomic_constants()

    @property
    @show_unit
    def k_alpha_2(self):
        try:
            val = Quantity(float(self.edges.loc["K"] - self.edges.loc["L II"]), 'keV')
        except(KeyError):
            val = '-'
        else:
            val = Quantity(val, UnitSettings.UNIT_E)
        return val

    @property
    @show_unit
    def k_alpha_1(self):
        try:
            val = Quantity(float(self.edges.loc["K"] - self.edges.loc["L III"]), 'keV')
        except(KeyError):
            val = '-'
        else:
            val = Quantity(val, UnitSettings.UNIT_E)
        return val

    @property
    def k_alpha(self):
        return (2 * self.k_alpha_1 + self.k_alpha_1) / 3

    @property
    @show_unit
    def k_beta(self):
        try:
            val = Quantity(float(self.edges.loc["K"] - self.edges.loc["M II"]), "keV")
        except(KeyError):
            val = '-'
        else:
            val = Quantity(val, UnitSettings.UNIT_E)
        return val

    @property
    def name(self):
        val = dabax.get_entry(self.symbol, "name")
        return val

    @property
    def neutron_booklet(self):
        ans = self._get_dabax_SLCS_DataBooklet()
        ans.columns = ['SCLC Data Booklet']
        return ans

    @property
    def neutron_news(self):
        ans = self._get_dabax_SLCS_NeutronNews()
        ans.columns = ['SCLC Neutron News']
        return ans

    @property
    def neutron_nist(self):
        ans = self._get_nist_b_sears()
        ans.columns = ['Nist Sears']
        return ans

    @property
    @show_unit
    def neutron_b_coh(self):
        try:
            ans = self._get_nist_b_sears().loc["Coh b (fm)"][0]
        except KeyError:
            return '-'
        if ans == '---':
            ans = '-'
        else:
            _ = complex(ans.replace('i', 'j'))
            ans = u.Quantity(_, 'fm')
        return ans

    @property
    @show_unit
    def neutron_b_inc(self):
        try:
            ans = self._get_nist_b_sears().loc["Inc b (fm)"][0]
        except KeyError:
            return '-'
        if ans == '---':
            ans = '-'
        else:
            _ = complex(ans.replace('i', 'j'))
            ans = u.Quantity(_, 'fm')
        return ans

    @property
    @show_unit
    def neutron_xs_coh(self):
        try:
            ans = self._get_nist_b_sears().loc["Coh xs (barn)"][0]
        except KeyError:
            return '-'
        return Quantity(ans, 'barn') if not ans == '---' else '-'

        # df[]) * unit if not df["Coh xs (barn)"][0] == '---' else np.nan,

    @property
    @show_unit
    def neutron_xs_inc(self):
        try:
            ans = self._get_nist_b_sears().loc["Inc xs (barn)"][0]
        except KeyError:
            return '-'
        return Quantity(ans, 'barn') if not ans == '---' else '-'

    @property
    @show_unit
    def neutron_abs_at_2200mps(self):
        try:
            ans = self._get_nist_b_sears().loc["Abs xs at 2200m/s (barn)"][0]
        except KeyError:
            return '-'
        return Quantity(ans, 'barn') if not ans == '---' else '-'

    @property
    def mup(self):
        return self.get_mup(self._energy)

    def get_mup(self, energy):
        return self._get_nist_mup_chantler(energy)

    def _get_nist_mup_chantler(self, energy):
        mu = self._get_nist_f1f2mu_chantler(energy)['µ/p total (cm2/g)']
        return Quantity(float(mu), 'cm^2/g')

    def _get_dabax_f0_waaskirf(self, q):
        df = dabax.get_table(self.symbol, "dabax_f0_WaasKirf")
        return Element.calc_dabax(q, df)

    def _get_dabax_f0_intertables(self, q):
        df = dabax.get_table(self.symbol, "dabax_f0_InterTables")
        return Element.calc_dabax(q, df, d=4)

    def _get_dabax_isf_balyuzi(self, q):
        df = dabax.get_table(self.element_symbol, "dabax_isf_Balyuzi")
        return Element.calc_dabax(q, df)

    def get_f0(self, q):
        try:
            return self._get_dabax_f0_waaskirf(q)
        except KeyError:
            return self._get_dabax_f0_intertables(q)

    def get_isf(self, q):
        return self._get_dabax_isf_balyuzi(q)

    def get_compton(self, energy, q):
        q = Quantity(q, "1/Å")
        _ = Quantity(energy, "keV")
        res1 = []
        res2 = []
        for energy in _:
            ttheta = Element.calc_ttheta(energy, q)

            crossec, energy_out = Element.crossec_compton_kleinnishina(energy, ttheta)
            isf = self.get_isf(q)
            res1.append(crossec * (self.atomic_number - isf))
            res2.append(energy_out)
        return np.array(res1), Quantity(res2)

    def get_thomson(self, energy, q):
        q = Quantity(q, "1/Å")
        energy = Quantity(energy, "keV")
        ttheta = Element.calc_ttheta(energy, q)

        res = (0.5 * (1 + np.cos(ttheta) ** 2) * abs(self.get_f(energy, q)) ** 2)

        return res

    def _get_nist_f1f2_chantler(self, energy):

        df = dabax.get_table(self.element_symbol, "nist_f1f2_chantler")

        rel_corr = float(dabax.get_entry(
            self.symbol,
            ["nist_f1f2_chantler", "relativistic_correction"],
        )[1])

        nt_corr = float(dabax.get_entry(
            self.symbol,
            ["nist_f1f2_chantler", "thomson_correction"]))

        res = []
        for e in energy:
            f1, f2 = Element.interpolate_chantler(e, df)
            res.append(f1 + rel_corr + nt_corr + 1j * f2)

        return res

    def _get_nist_f1f2mu_chantler(self, energy):
        """
        dabax:f1f2_Chantler
        """
        df = dabax.get_table(self.element_symbol, "nist_f1f2_chantler")
        energy = Quantity(energy, 'keV').value
        nist = df.values.astype(float).T
        arr = np.abs(nist[0] - energy)
        idx = arr.argsort()[:2]  # find nearest data-points
        out = []
        for i in arange(1, 8):
            p1 = poly1d(
                polyfit(
                    nist[0, idx],
                    nist[i, idx],
                    1,
                )
            )
            out.append(p1(energy))

        out.append(
            dabax.get_entry(
                self.symbol,
                ["nist_f1f2_chantler", "relativistic_correction"],
            )
        )
        out.append(
            dabax.get_entry(
                self.symbol,
                ["nist_f1f2_chantler", "thomson_correction"],
            )
        )
        result = DataFrame(
            [out],
            columns=[
                "f1 (e/atom)",
                "f2 (e/atom)",
                "µ/p photoel (cm2/g)",
                "µ/p coh+inc (cm2/g)",
                "µ/p total (cm2/g)",
                "µ/p k (cm2/g)",
                "lambda (nm)",
                "relativistic_correction",
                "thomson_correction",
            ],
        )
        return result

    def _get_cxro_f1f2_henke(self, energy, degree=1, interpolation_delta_idx=1):
        """
        dabax: f1f2_Henke
        """
        _energy = energy
        res = []
        for energy in _energy:
            energy *= 1e3
            df = dabax.get_table(self.element_symbol, "cxro_f1f2_henke")
            e = df["E (eV)"].values
            f1 = df["f1"].values
            f2 = df["f2"].values
            idx = argmax(e >= energy)
            p1 = poly1d(
                polyfit(
                    e[idx - interpolation_delta_idx: idx + interpolation_delta_idx],
                    f1[idx - interpolation_delta_idx: idx + interpolation_delta_idx],
                    degree,
                )
            )
            p2 = poly1d(
                polyfit(
                    e[idx - interpolation_delta_idx: idx + interpolation_delta_idx],
                    f2[idx - interpolation_delta_idx: idx + interpolation_delta_idx],
                    degree,
                )
            )
            res.append(p1(energy) + 1j * p2(energy))
        return res

    def _get_dabax_SLCS_DataBooklet(self):
        df = dabax.get_table(self.symbol, "dabax_Neutron_SLCS_DataBooklet")
        return df.T

    def _get_dabax_SLCS_NeutronNews(self):
        df = dabax.get_table(self.symbol, "dabax_Neutron_SLCS_NeutronNews")
        return df.T

    def _get_nist_b_sears(self):
        df = dabax.get_table(self.symbol, "nist_b_sears")
        return df.T

    def _get_f1f2(self, energy, databank="auto"):
        if databank == "auto":
            databank = "cxro" if energy.max() <= 30 else "nist"

        if databank in ["CXRO", "cxro", "henke", "Henke"]:
            return self._get_cxro_f1f2_henke(energy)
        if databank in ["NIST", "nist", "chantler", "Chantler"]:
            return self._get_nist_f1f2_chantler(energy)

    def get_f(self, energy, q, **params):
        q = Quantity(q, UnitSettings.UNIT_Q)
        energy = Quantity(energy, UnitSettings.UNIT_E)

        q = Quantity(q, "1/Å").value  # Ensure parameters are in right unit for table
        energy = Quantity(energy, "keV").value

        f = self._get_f1f2(energy, **params)
        f0 = self.get_f0(q)
        x = np.tile(f, (len(f0), 1)).T
        y = np.tile(f0, (len(f), 1))

        res = x + y - self.atomic_number
        return res

    def _get_nist_edges_chantler(self):
        # always from element
        df = dabax.get_table(self.symbol, "nist_edges_chantler")

        df['E (keV)'] = Quantity(df['E (keV)'].values, 'keV').to(UnitSettings.UNIT_E).value
        df.columns = ['E ({:s})'.format(UnitSettings.UNIT_E)]
        return df

    def _get_dabax_atomic_constants(self):
        df = dabax.get_table(self.symbol, 'dabax_AtomicConstants').T
        df.columns = ['values']
        return df


class Ion(Element):

    def _repr_html_(self):

        try:
            ed = self.edges._repr_html_()
        except:
            ed = ''

        keys = [["Symbol",
                 "%s<sup>%s</sup>" % (self.element_symbol, (str(abs(self.charge)) + "-" if self.charge < 0 else str(
                     self.charge) + "+"))],
                ["Atomic number", self.atomic_number],
                ["Atomic mass", self.atomic_weight],
                ["Charge", self.charge],
                ["Melting point", self.melting_point],
                ["Boiling point", self.boiling_point],
                ["Energy", self._energy[0]],
                ["q", self._q[0]],
                ["2θ", self._ttheta().flat[0]],
                ["X-ray formfactor", '-' if self.f.flat[0] == '-' else self.f.flat[0].real * u.electron],
                ["K<sub>α1</sub>", self.k_alpha_1],
                ["K<sub>α2</sub>", self.k_alpha_2],
                ["K<sub>β</sub>", self.k_beta],
                ["b<sub>coh</sub>", self.neutron_b_coh],
                ["b<sub>inc</sub>", self.neutron_b_inc],
                ["σ<sub>coh</sub>", self.neutron_xs_coh],
                ["σ<sub>inc</sub>", self.neutron_xs_inc],
                ["absorption (2200m/s)", self.neutron_abs_at_2200mps]]
        tab1 = ["<tr> <th>{}</th> <td>{:}</td> </tr>".format(name,
                                                             value if
                                                             isinstance(value, (str, int)) else value.round(4)) for
                name, value in keys]
        ans = ("<h1>{}({})</h1>".format(self.name,
                                        (str(abs(self.charge)) + "-" if self.charge < 0 else str(
                                            self.charge) + "+")) +
               ("<table> " +
                "".join(tab1) +
                "</table>") +
               ed)
        return ans

    def _get_nist_edges_chantler(self):
        # always from element
        # warnings.warn("Edges from Element")
        df = dabax.get_table(self.element_symbol, "nist_edges_chantler")

        df['E (keV)'] = Quantity(df['E (keV)'].values, 'keV').to(UnitSettings.UNIT_E).value
        df.columns = ['E ({:s})'.format(UnitSettings.UNIT_E)]
        return df

    def _get_dabax_SLCS_DataBooklet(self):
        df = dabax.get_table(self.element_symbol, "dabax_Neutron_SLCS_DataBooklet")
        return df.T

    def _get_dabax_SLCS_NeutronNews(self):
        df = dabax.get_table(self.element_symbol, "dabax_Neutron_SLCS_NeutronNews")
        return df.T

    def _get_nist_b_sears(self):
        df = dabax.get_table(self.element_symbol, "nist_b_sears")
        return df.T

    def _get_dabax_atomic_constants(self):
        df = dabax.get_table(self.element_symbol, 'dabax_AtomicConstants').T
        df.columns = ['values']
        return df

    @property
    def mcgowan_volume(self):
        val = dabax.get_entry(self.element_symbol, "mcgowan_vol")
        return Quantity(val, 'Å^3')


class Isotope(Element):

    def _repr_html_(self):

        try:
            ed = self.edges._repr_html_()
        except:
            ed = ''

        try:
            nn = self.neutron_booklet._repr_html_()
        except:
            nn = ''

        keys = [["Symbol", "<sup>%s</sup>%s" % (self.mass_number, self.element_symbol)],
                ["Atomic number", self.atomic_number],
                ["Atomic mass", self.atomic_weight],
                ["Charge", self.charge],
                ["Atomic radius", self.atomic_radius],
                ["Covalent radius", self.covalent_radius],
                ["Melting point", self.melting_point],
                ["Boiling point", self.boiling_point],

                ["Energy", self._energy[0]],
                ["q", self._q[0]],
                ["2θ", self._ttheta().flat[0]],
                ["X-ray formfactor", '-' if self.f.flat[0] == '-' else self.f.flat[0].real * u.electron],
                ["K<sub>α1</sub>", self.k_alpha_1],
                ["K<sub>α2</sub>", self.k_alpha_2],
                ["K<sub>β</sub>", self.k_beta],
                ["b<sub>coh</sub>", self.neutron_b_coh],
                ["b<sub>inc</sub>", self.neutron_b_inc],
                ["σ<sub>coh</sub>", self.neutron_xs_coh],
                ["σ<sub>inc</sub>", self.neutron_xs_inc],
                ["absorption (2200m/s)", self.neutron_abs_at_2200mps]]

        tab1 = ["<tr> <th>{}</th> <td>{:}</td> </tr>".format(name,
                                                             value if isinstance(value,
                                                                                 (str, int)) else value.round(4))
                for name, value in keys]
        ans = ("<h1>{}</h1>".format(self.name,
                                    (str(abs(self.charge)) + "-" if self.charge < 0 else str(
                                        self.charge) + "+")) +
               ("<table> " +
                "".join(tab1) +
                "</table>") +
               ed + nn)
        return ans

    def _get_dabax_f0_waaskirf(self, q):
        df = dabax.get_table(self.element_symbol, "dabax_f0_WaasKirf")
        return Element.calc_dabax(q, df)

    def _get_nist_edges_chantler(self):
        df = dabax.get_table(self.element_symbol, "nist_edges_chantler")

        df['E (keV)'] = Quantity(df['E (keV)'].values, 'keV').to(UnitSettings.UNIT_E).value
        df.columns = ['E ({:s})'.format(UnitSettings.UNIT_E)]
        return df

    @property
    def mcgowan_volume(self):
        val = dabax.get_entry(self.element_symbol, "mcgowan_vol")
        return Quantity(val, 'Å^3')


class ElementCounter:
    def __init__(self):
        self.elements = ElementsHill.copy()

    def add(self, comp):
        if isinstance(comp, ElementCounter):
            comp = comp.get_elements()
            for k in comp:
                self.elements[k] += comp[k]
        else:
            symbol, number = comp

            self.elements[symbol] += number

    def mul(self, mul):
        for k in self.elements:
            self.elements[k] *= mul

    def get_elements(self):
        d = self.elements
        return {k: d[k] for k in d if not d[k] == 0}


class Compound(BaseElement):
    def __init__(self, formula, energy='8.047 keV', q='0 1/Å', density='mcgowan'):
        super().__init__(q, energy)
        self.formula = formula
        self.composition = formula
        if density == 'mcgowan':
            self.density = self._guess_density_mcgowan()
        elif density == 'element':
            self.density = self._guess_density_element()
        else:
            self.density = density

    def __repr__(self):
        return str(self.formula) + '\n' + self.molecular_weight.__str__() + '\n' + str(self.composition)

    # def _repr_latex_(self) -> str:
    #
    #    return r"$\mathrm{{{}}}$".format(self.formula) + \
    #           r"$\\ \mathrm{m_w}:\,$" + self.molecular_weight._repr_latex_() + \
    #           r"$\\ \rho:\,$" + self.density._repr_latex_()

    @property
    def composition_table(self):
        y = DataFrame(self.composition, index=['x']).T
        res = []
        for x in y.T:
            res.append(Elements[x].molar_mass)
        y['molar mass'] = Quantity(res, 'g/mol').value
        y['comp %'] = y['x'].values / y['x'].sum() * 100
        mass = y['x'] * y['molar mass']
        y['mass %'] = mass / mass.sum() * 100
        return y

    @property
    def electron_number(self):
        res = 0
        for k in self.composition:
            res += float(Elements[k].electron_number * self.composition[k])
        return res

    @property
    def composition(self):
        return self._composition

    @composition.setter
    def composition(self, formula):
        self._lst, comp = Compound.parse_formula(formula)
        self._composition = comp.get_elements()

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, value):
        self._density = Quantity(value, 'g/cm^3')

    @property
    def f(self):
        res = np.zeros([len(self._energy), len(self._q)], dtype=np.complex128)
        for k in self.composition:
            res += Elements[k].get_f(self._energy, self._q) * self.composition[k]
        return res

    def _guess_density_mcgowan(self):
        V = 0
        m = 0
        try:
            for k in self.composition:
                V += Elements[k].mcgowan_volume * self.composition[k]
                m += Elements[k].molar_mass * self.composition[k]
        except KeyError:
            return u.Quantity('1 g/cm^3')
        return (m / V / c.N_A).to('g/cm^3')


    def _guess_density_element(self):
        V = 0
        m = 0
        try:
            for k in self.composition:
                V += Elements[k].molar_mass / Elements[k].density / c.N_A * self.composition[k]
                m += Elements[k].molar_mass * self.composition[k]
        except KeyError:
            return u.Quantity('1 g/cm^3')
        return (m / V / c.N_A).to('g/cm^3')

    @property
    def crossec_thomson(self):
        res = np.zeros_like(self.q.value, dtype=np.complex128)
        for k in self.composition:
            res += (Elements[k].get_thomson(self.energy, self.q)).value * self.composition[k]
        return res / self.n

    @property
    def crossec_thomson_sq(self):

        ttheta = Compound.calc_ttheta(self.energy, self.q)
        res = np.zeros_like(self.q.value, dtype=np.complex128)
        for k in self.composition:
            res += Elements[k].get_f(self.energy, self.q) * self.composition[k]
        return 0.5 * (1 + np.cos(ttheta) ** 2) * abs(res) ** 2 / self.n ** 2

    @property
    def crossec_compton(self):
        res = np.zeros_like(self.q.value)
        for k in self.composition:
            res += (Elements[k].get_compton(self.energy, self.q)[0]).value * self.composition[k]
        return res / self.n, Elements[k].get_compton(self.energy, self.q)[1]

    @property
    def n(self):
        res = 0
        for k in self.composition:
            res += self.composition[k]
        return res

    @property
    def mu(self):
        return self.mup * self.density

    def get_mu(self, energy):
        return self.get_mup(energy) * self.density

    def get_mup(self, energy):
        res = 0
        for k in self.composition:
            res += Elements[k].get_mup(energy) * Elements[k].atomic_weight * self.composition[k]
        return res / self.molecular_weight

    @property
    def mup(self):
        return self.get_mup(self.energy)

    @property
    def molecular_weight(self):
        mw = 0
        for k in self.composition:
            mw += Elements[k].molar_mass * self.composition[k]
        return Quantity(mw, 'g/mol')

    @property
    @show_unit
    def q_crit_sq(self):
        qcsq = 16 * np.pi * r_e * self.f / self.molecular_volume
        return Quantity(np.squeeze(qcsq), '1/Å^2')

    def _q_crit(self):
        qcsq = 16 * np.pi * r_e * self.f / self.molecular_volume
        return Quantity(qcsq ** .5, '1/Å')

    @property
    @show_unit
    def q_crit(self):
        return np.squeeze(self._q_crit())

    @property
    def ttheta_crit(self):
        ans = []
        for e, q in zip(self._energy, self._q_crit()):
            ans.append(self.calc_ttheta(e, q))
        ans = u.Quantity(ans, UnitSettings.UNIT_TTH)
        return np.squeeze(ans)

    @property
    def deltabeta(self):

        db = self.wavelength ** 2 / (2 * np.pi) * r_e / self.molecular_volume * self.f
        return db

    @property
    def molecular_volume(self):
        mv = self.molecular_weight / self.density / c.N_A
        return Quantity(mv, 'Å^3')

    def _repr_html_(self):

        keys = [
            ["Composition",
             self.composition_table.round({'x': 3, 'molar mass': 3, 'comp %': 1, 'mass %': 1})._repr_html_()],
            ["Molecular Weight", self.molecular_weight],
            ["Molecular Volume", self.molecular_volume],
            ["Density", self.density],
            ["q<sub>crit</sub>", self.q_crit.flat[0].real],
            ["θ<sub>crit</sub>", self.ttheta_crit.flat[0].real / 2],
            ["δ", "%.3e" % self.deltabeta.flat[0].real],
            ["iβ", "i%.3e" % self.deltabeta.flat[0].imag],
            ["Energy", self._energy[0]],
            ["q", self._q[0]],
            ["2θ", self._ttheta().flat[0]],
            ["X-ray formfactor", '-' if self.f.flat[0] == '-' else self.f.flat[0].real * u.electron],
            ["ρ<sub>xff</sub>",
             '-' if self.f.flat[0] == '-' else self.f.flat[0].real * u.electron / self.molecular_volume],
            ["r<sub>e</sub>ρ<sub>xff</sub>",
             '-' if self.f.flat[0] == '-' else '%s'%(self.f.flat[0].real / self.molecular_volume * r_e).to('10^10/cm^2').round(4)],


        ]

        tab1 = ["<tr> <th>{}</th> <td>{:}</td> </tr>".format(name,
                                                             value if isinstance(value,
                                                                                 (
                                                                                     str, int, dict)) else value.round(
                                                                 2))
                for name, value in keys]
        ans = ("<h1>{}</h1>".format(self.formula_to_str(self._lst)) +

               "<table> " +
               "".join(tab1) +
               "</table>")

        return ans


    @staticmethod
    def parse_string_to_list(element_str):
        res = []
        # find all possible sub formulas
        match = re.findall(
            # Element-Charge-Number | Element-Number | (Element OR subformula)-number
            r"\d*[A-Z][a-z]?\d?[+|-]\d*(?:\.\d+)?|\d*[A-Z][a-z]?\d*(?:\.\d+)?|\((?:[^()]*(?:\(.*\))?[^()]*)+\)\d+(?:\.\d+)?",
            element_str,
        )
        if match == [element_str]:
            submatch = re.match(r"(\d*[A-Z][a-z]?(?:\d?[+|-])?)(\d*(?:\.\d+)?)", element_str)
            if submatch:
                e = submatch.group(1)
                n = (
                    1
                    if submatch.group(2) == ""
                    else (
                        int(submatch.group(2))
                        if float(submatch.group(2)).is_integer()
                        else float(submatch.group(2))
                    )
                )
                res.append([e, n])
            else:
                #print(element_str)
                subsub = re.match(r"\((.*)\)(\d*(?:\.\d+)?)", element_str)
                nn = (
                    1
                    if subsub.group(2) == ""
                    else (
                        int(subsub.group(2))
                        if float(subsub.group(2)).is_integer()
                        else float(subsub.group(2))
                    )
                )
                ll = Compound.parse_string_to_list(subsub.group(1))
                res.append([ll, nn])
        else:
            for m in match:
                r = Compound.parse_string_to_list(m)
                res.append(r[0])
        return res


    @staticmethod
    def parse_list_to_counter(comp):
        counter = ElementCounter()
        for subunit in comp:
            if isinstance(subunit[1], (int, float)):
                element, number = subunit

                if isinstance(element, str):
                    counter.add(subunit)

                else:
                    counter_sub = Compound.parse_list_to_counter([element])
                    counter_sub.mul(number)
                    counter.add(counter_sub)
            else:
                counter_sub = Compound.parse_list_to_counter(subunit)
                counter.add(counter_sub)

        return counter


    @staticmethod
    def parse_formula(formula):
        lst = Compound.parse_string_to_list(formula)
        counter = Compound.parse_list_to_counter(lst)
        return lst, counter


    @staticmethod
    def formula_to_str(lst):
        try:
            e, n = lst
            if isinstance(e, list) and isinstance(n, list):
                raise Exception('Error')
            if not isinstance(e, list):
                x = re.match('(\d*\.?\d*)([A-Z][a-z]?)(\d+\+|\d+\-|\+|\-)?', e)
                iso = '<sup>%s</sup>' % x.group(1) if not x.group(1) == '' else ''
                if x.group(3):
                    charge = '<sup>%s</sup>' % x.group(3)
                else:
                    charge = ''
                e = iso + x.group(2) + charge
                ans = '%s<sub>%g</sub>' % (e, n) if not n == 1 else '%s' % e
                return ans
            else:
                ans = []
                for y in e:
                    ret = Compound.formula_to_str(y)
                    ans.append(ret)
                return '(%s)<sub>%g</sub>' % (''.join(ans), n)
        except:
            ans = []
            for x in lst:
                ans.append(Compound.formula_to_str(x))
            return ''.join(ans)


    def __add__(self, otherCompound):
        V =  self.molecular_volume + otherCompound.molecular_volume
        m = self.molecular_weight + otherCompound.molecular_weight
        formula = self.formula + otherCompound.formula
        newCp = Compound(formula, density=m/V/c.N_A)

        return newCp


    def __mul__(self, multiplier):
        formula = "(%s)%g"%(self.formula, multiplier)
        newCp = Compound(formula, density = self.density)

        return newCp


    def __rmul__(self, multiplier):
        formula = "(%s)%g"%(self.formula, multiplier)
        newCp = Compound(formula, density = self.density)

        return newCp



Elements = {
    "H": Element("H"),
    "He": Element("He"),
    "Li": Element("Li"),
    "Be": Element("Be"),
    "B": Element("B"),
    "C": Element("C"),
    "N": Element("N"),
    "O": Element("O"),
    "F": Element("F"),
    "Ne": Element("Ne"),
    "Na": Element("Na"),
    "Mg": Element("Mg"),
    "Al": Element("Al"),
    "Si": Element("Si"),
    "P": Element("P"),
    "S": Element("S"),
    "Cl": Element("Cl"),
    "Ar": Element("Ar"),
    "K": Element("K"),
    "Ca": Element("Ca"),
    "Sc": Element("Sc"),
    "Ti": Element("Ti"),
    "V": Element("V"),
    "Cr": Element("Cr"),
    "Mn": Element("Mn"),
    "Fe": Element("Fe"),
    "Co": Element("Co"),
    "Ni": Element("Ni"),
    "Cu": Element("Cu"),
    "Zn": Element("Zn"),
    "Ga": Element("Ga"),
    "Ge": Element("Ge"),
    "As": Element("As"),
    "Se": Element("Se"),
    "Br": Element("Br"),
    "Kr": Element("Kr"),
    "Rb": Element("Rb"),
    "Sr": Element("Sr"),
    "Y": Element("Y"),
    "Zr": Element("Zr"),
    "Nb": Element("Nb"),
    "Mo": Element("Mo"),
    "Tc": Element("Tc"),
    "Ru": Element("Ru"),
    "Rh": Element("Rh"),
    "Pd": Element("Pd"),
    "Ag": Element("Ag"),
    "Cd": Element("Cd"),
    "In": Element("In"),
    "Sn": Element("Sn"),
    "Sb": Element("Sb"),
    "Te": Element("Te"),
    "I": Element("I"),
    "Xe": Element("Xe"),
    "Cs": Element("Cs"),
    "Ba": Element("Ba"),
    "La": Element("La"),
    "Ce": Element("Ce"),
    "Pr": Element("Pr"),
    "Nd": Element("Nd"),
    "Pm": Element("Pm"),
    "Sm": Element("Sm"),
    "Eu": Element("Eu"),
    "Gd": Element("Gd"),
    "Tb": Element("Tb"),
    "Dy": Element("Dy"),
    "Ho": Element("Ho"),
    "Er": Element("Er"),
    "Tm": Element("Tm"),
    "Yb": Element("Yb"),
    "Lu": Element("Lu"),
    "Hf": Element("Hf"),
    "Ta": Element("Ta"),
    "W": Element("W"),
    "Re": Element("Re"),
    "Os": Element("Os"),
    "Ir": Element("Ir"),
    "Pt": Element("Pt"),
    "Au": Element("Au"),
    "Hg": Element("Hg"),
    "Tl": Element("Tl"),
    "Pb": Element("Pb"),
    "Bi": Element("Bi"),
    "Po": Element("Po"),
    "At": Element("At"),
    "Rn": Element("Rn"),
    "Fr": Element("Fr"),
    "Ra": Element("Ra"),
    "Ac": Element("Ac"),
    "Th": Element("Th"),
    "Pa": Element("Pa"),
    "U": Element("U"),
    "Np": Element("Np"),
    "Pu": Element("Pu"),
    "Am": Element("Am"),
    "Cm": Element("Cm"),
    "Bk": Element("Bk"),
    "Cf": Element("Cf"),
    "Es": Element("Es"),
    "Fm": Element("Fm"),
    "Md": Element("Md"),
    "No": Element("No"),
    "Lr": Element("Lr"),
    "Rf": Element("Rf"),
    "Db": Element("Db"),
    "Sg": Element("Sg"),
    "Bh": Element("Bh"),
    "Hs": Element("Hs"),
    "Mt": Element("Mt"),
    "Ds": Element("Ds"),
    "Rg": Element("Rg"),
    "Cn": Element("Cn"),
    "Nh": Element("Nh"),
    "Fl": Element("Fl"),
    "Mc": Element("Mc"),
    "Lv": Element("Lv"),
    "Ts": Element("Ts"),
    "Og": Element("Og"),
    "1H": Isotope("1H"),
    "2H": Isotope("2H"),
    "3H": Isotope("3H"),
    "3He": Isotope("3He"),
    "4He": Isotope("4He"),
    "6Li": Isotope("6Li"),
    "7Li": Isotope("7Li"),
    "9Be": Isotope("9Be"),
    "10B": Isotope("10B"),
    "11B": Isotope("11B"),
    "12C": Isotope("12C"),
    "13C": Isotope("13C"),
    "14C": Isotope("14C"),
    "14N": Isotope("14N"),
    "15N": Isotope("15N"),
    "16O": Isotope("16O"),
    "17O": Isotope("17O"),
    "18O": Isotope("18O"),
    "19F": Isotope("19F"),
    "20Ne": Isotope("20Ne"),
    "21Ne": Isotope("21Ne"),
    "22Ne": Isotope("22Ne"),
    "23Na": Isotope("23Na"),
    "24Mg": Isotope("24Mg"),
    "25Mg": Isotope("25Mg"),
    "26Mg": Isotope("26Mg"),
    "27Al": Isotope("27Al"),
    "28Si": Isotope("28Si"),
    "29Si": Isotope("29Si"),
    "30Si": Isotope("30Si"),
    "31P": Isotope("31P"),
    "32S": Isotope("32S"),
    "33S": Isotope("33S"),
    "34S": Isotope("34S"),
    "36S": Isotope("36S"),
    "35Cl": Isotope("35Cl"),
    "37Cl": Isotope("37Cl"),
    "36Ar": Isotope("36Ar"),
    "38Ar": Isotope("38Ar"),
    "40Ar": Isotope("40Ar"),
    "39K": Isotope("39K"),
    "40K": Isotope("40K"),
    "41K": Isotope("41K"),
    "40Ca": Isotope("40Ca"),
    "42Ca": Isotope("42Ca"),
    "43Ca": Isotope("43Ca"),
    "44Ca": Isotope("44Ca"),
    "46Ca": Isotope("46Ca"),
    "48Ca": Isotope("48Ca"),
    "45Sc": Isotope("45Sc"),
    "46Ti": Isotope("46Ti"),
    "47Ti": Isotope("47Ti"),
    "48Ti": Isotope("48Ti"),
    "49Ti": Isotope("49Ti"),
    "50Ti": Isotope("50Ti"),
    "50V": Isotope("50V"),
    "51V": Isotope("51V"),
    "50Cr": Isotope("50Cr"),
    "52Cr": Isotope("52Cr"),
    "53Cr": Isotope("53Cr"),
    "54Cr": Isotope("54Cr"),
    "54Fe": Isotope("54Fe"),
    "56Fe": Isotope("56Fe"),
    "57Fe": Isotope("57Fe"),
    "58Fe": Isotope("58Fe"),
    "55Mn": Isotope("55Mn"),
    "58Ni": Isotope("58Ni"),
    "60Ni": Isotope("60Ni"),
    "61Ni": Isotope("61Ni"),
    "62Ni": Isotope("62Ni"),
    "64Ni": Isotope("64Ni"),
    "59Co": Isotope("59Co"),
    "63Cu": Isotope("63Cu"),
    "65Cu": Isotope("65Cu"),
    "64Zn": Isotope("64Zn"),
    "66Zn": Isotope("66Zn"),
    "67Zn": Isotope("67Zn"),
    "68Zn": Isotope("68Zn"),
    "70Zn": Isotope("70Zn"),
    "69Ga": Isotope("69Ga"),
    "71Ga": Isotope("71Ga"),
    "70Ge": Isotope("70Ge"),
    "72Ge": Isotope("72Ge"),
    "73Ge": Isotope("73Ge"),
    "74Ge": Isotope("74Ge"),
    "76Ge": Isotope("76Ge"),
    "74Se": Isotope("74Se"),
    "76Se": Isotope("76Se"),
    "77Se": Isotope("77Se"),
    "78Se": Isotope("78Se"),
    "80Se": Isotope("80Se"),
    "82Se": Isotope("82Se"),
    "75As": Isotope("75As"),
    "79Br": Isotope("79Br"),
    "81Br": Isotope("81Br"),
    "78Kr": Isotope("78Kr"),
    "80Kr": Isotope("80Kr"),
    "82Kr": Isotope("82Kr"),
    "83Kr": Isotope("83Kr"),
    "84Kr": Isotope("84Kr"),
    "86Kr": Isotope("86Kr"),
    "85Rb": Isotope("85Rb"),
    "87Rb": Isotope("87Rb"),
    "84Sr": Isotope("84Sr"),
    "86Sr": Isotope("86Sr"),
    "87Sr": Isotope("87Sr"),
    "88Sr": Isotope("88Sr"),
    "89Y": Isotope("89Y"),
    "90Zr": Isotope("90Zr"),
    "91Zr": Isotope("91Zr"),
    "92Zr": Isotope("92Zr"),
    "93Zr": Isotope("93Zr"),
    "94Zr": Isotope("94Zr"),
    "96Zr": Isotope("96Zr"),
    "93Nb": Isotope("93Nb"),
    "92Mo": Isotope("92Mo"),
    "94Mo": Isotope("94Mo"),
    "95Mo": Isotope("95Mo"),
    "96Mo": Isotope("96Mo"),
    "97Mo": Isotope("97Mo"),
    "98Mo": Isotope("98Mo"),
    "100Mo": Isotope("100Mo"),
    "97Tc": Isotope("97Tc"),
    "98Tc": Isotope("98Tc"),
    "99Tc": Isotope("99Tc"),
    "96Ru": Isotope("96Ru"),
    "98Ru": Isotope("98Ru"),
    "99Ru": Isotope("99Ru"),
    "100Ru": Isotope("100Ru"),
    "101Ru": Isotope("101Ru"),
    "102Ru": Isotope("102Ru"),
    "104Ru": Isotope("104Ru"),
    "103Rh": Isotope("103Rh"),
    "102Pd": Isotope("102Pd"),
    "104Pd": Isotope("104Pd"),
    "105Pd": Isotope("105Pd"),
    "106Pd": Isotope("106Pd"),
    "108Pd": Isotope("108Pd"),
    "110Pd": Isotope("110Pd"),
    "107Ag": Isotope("107Ag"),
    "109Ag": Isotope("109Ag"),
    "106Cd": Isotope("106Cd"),
    "108Cd": Isotope("108Cd"),
    "110Cd": Isotope("110Cd"),
    "111Cd": Isotope("111Cd"),
    "112Cd": Isotope("112Cd"),
    "113Cd": Isotope("113Cd"),
    "114Cd": Isotope("114Cd"),
    "116Cd": Isotope("116Cd"),
    "113In": Isotope("113In"),
    "115In": Isotope("115In"),
    "112Sn": Isotope("112Sn"),
    "114Sn": Isotope("114Sn"),
    "115Sn": Isotope("115Sn"),
    "116Sn": Isotope("116Sn"),
    "117Sn": Isotope("117Sn"),
    "118Sn": Isotope("118Sn"),
    "119Sn": Isotope("119Sn"),
    "120Sn": Isotope("120Sn"),
    "122Sn": Isotope("122Sn"),
    "124Sn": Isotope("124Sn"),
    "121Sb": Isotope("121Sb"),
    "123Sb": Isotope("123Sb"),
    "120Te": Isotope("120Te"),
    "122Te": Isotope("122Te"),
    "123Te": Isotope("123Te"),
    "124Te": Isotope("124Te"),
    "125Te": Isotope("125Te"),
    "126Te": Isotope("126Te"),
    "128Te": Isotope("128Te"),
    "130Te": Isotope("130Te"),
    "127I": Isotope("127I"),
    "124Xe": Isotope("124Xe"),
    "126Xe": Isotope("126Xe"),
    "128Xe": Isotope("128Xe"),
    "129Xe": Isotope("129Xe"),
    "130Xe": Isotope("130Xe"),
    "131Xe": Isotope("131Xe"),
    "132Xe": Isotope("132Xe"),
    "134Xe": Isotope("134Xe"),
    "136Xe": Isotope("136Xe"),
    "133Cs": Isotope("133Cs"),
    "130Ba": Isotope("130Ba"),
    "132Ba": Isotope("132Ba"),
    "134Ba": Isotope("134Ba"),
    "135Ba": Isotope("135Ba"),
    "136Ba": Isotope("136Ba"),
    "137Ba": Isotope("137Ba"),
    "138Ba": Isotope("138Ba"),
    "138La": Isotope("138La"),
    "139La": Isotope("139La"),
    "136Ce": Isotope("136Ce"),
    "138Ce": Isotope("138Ce"),
    "140Ce": Isotope("140Ce"),
    "142Ce": Isotope("142Ce"),
    "141Pr": Isotope("141Pr"),
    "142Nd": Isotope("142Nd"),
    "143Nd": Isotope("143Nd"),
    "144Nd": Isotope("144Nd"),
    "145Nd": Isotope("145Nd"),
    "146Nd": Isotope("146Nd"),
    "148Nd": Isotope("148Nd"),
    "150Nd": Isotope("150Nd"),
    "145Pm": Isotope("145Pm"),
    "147Pm": Isotope("147Pm"),
    "144Sm": Isotope("144Sm"),
    "147Sm": Isotope("147Sm"),
    "148Sm": Isotope("148Sm"),
    "149Sm": Isotope("149Sm"),
    "150Sm": Isotope("150Sm"),
    "152Sm": Isotope("152Sm"),
    "154Sm": Isotope("154Sm"),
    "151Eu": Isotope("151Eu"),
    "153Eu": Isotope("153Eu"),
    "152Gd": Isotope("152Gd"),
    "154Gd": Isotope("154Gd"),
    "155Gd": Isotope("155Gd"),
    "156Gd": Isotope("156Gd"),
    "157Gd": Isotope("157Gd"),
    "158Gd": Isotope("158Gd"),
    "160Gd": Isotope("160Gd"),
    "159Tb": Isotope("159Tb"),
    "156Dy": Isotope("156Dy"),
    "158Dy": Isotope("158Dy"),
    "160Dy": Isotope("160Dy"),
    "161Dy": Isotope("161Dy"),
    "162Dy": Isotope("162Dy"),
    "163Dy": Isotope("163Dy"),
    "164Dy": Isotope("164Dy"),
    "165Ho": Isotope("165Ho"),
    "162Er": Isotope("162Er"),
    "164Er": Isotope("164Er"),
    "166Er": Isotope("166Er"),
    "167Er": Isotope("167Er"),
    "168Er": Isotope("168Er"),
    "170Er": Isotope("170Er"),
    "169Tm": Isotope("169Tm"),
    "168Yb": Isotope("168Yb"),
    "170Yb": Isotope("170Yb"),
    "171Yb": Isotope("171Yb"),
    "172Yb": Isotope("172Yb"),
    "173Yb": Isotope("173Yb"),
    "174Yb": Isotope("174Yb"),
    "176Yb": Isotope("176Yb"),
    "175Lu": Isotope("175Lu"),
    "176Lu": Isotope("176Lu"),
    "174Hf": Isotope("174Hf"),
    "176Hf": Isotope("176Hf"),
    "177Hf": Isotope("177Hf"),
    "178Hf": Isotope("178Hf"),
    "179Hf": Isotope("179Hf"),
    "180Hf": Isotope("180Hf"),
    "180Ta": Isotope("180Ta"),
    "181Ta": Isotope("181Ta"),
    "180W": Isotope("180W"),
    "182W": Isotope("182W"),
    "183W": Isotope("183W"),
    "184W": Isotope("184W"),
    "186W": Isotope("186W"),
    "185Re": Isotope("185Re"),
    "187Re": Isotope("187Re"),
    "184Os": Isotope("184Os"),
    "186Os": Isotope("186Os"),
    "187Os": Isotope("187Os"),
    "188Os": Isotope("188Os"),
    "189Os": Isotope("189Os"),
    "190Os": Isotope("190Os"),
    "192Os": Isotope("192Os"),
    "191Ir": Isotope("191Ir"),
    "193Ir": Isotope("193Ir"),
    "190Pt": Isotope("190Pt"),
    "192Pt": Isotope("192Pt"),
    "194Pt": Isotope("194Pt"),
    "195Pt": Isotope("195Pt"),
    "196Pt": Isotope("196Pt"),
    "198Pt": Isotope("198Pt"),
    "197Au": Isotope("197Au"),
    "196Hg": Isotope("196Hg"),
    "198Hg": Isotope("198Hg"),
    "199Hg": Isotope("199Hg"),
    "200Hg": Isotope("200Hg"),
    "201Hg": Isotope("201Hg"),
    "202Hg": Isotope("202Hg"),
    "204Hg": Isotope("204Hg"),
    "203Tl": Isotope("203Tl"),
    "205Tl": Isotope("205Tl"),
    "204Pb": Isotope("204Pb"),
    "206Pb": Isotope("206Pb"),
    "207Pb": Isotope("207Pb"),
    "208Pb": Isotope("208Pb"),
    "209Bi": Isotope("209Bi"),
    "209Po": Isotope("209Po"),
    "210Po": Isotope("210Po"),
    "210At": Isotope("210At"),
    "211At": Isotope("211At"),
    "211Rn": Isotope("211Rn"),
    "220Rn": Isotope("220Rn"),
    "222Rn": Isotope("222Rn"),
    "223Fr": Isotope("223Fr"),
    "223Ra": Isotope("223Ra"),
    "224Ra": Isotope("224Ra"),
    "226Ra": Isotope("226Ra"),
    "228Ra": Isotope("228Ra"),
    "227Ac": Isotope("227Ac"),
    "230Th": Isotope("230Th"),
    "232Th": Isotope("232Th"),
    "231Pa": Isotope("231Pa"),
    "233U": Isotope("233U"),
    "234U": Isotope("234U"),
    "235U": Isotope("235U"),
    "236U": Isotope("236U"),
    "238U": Isotope("238U"),
    "237Np": Isotope("237Np"),
    "239Np": Isotope("239Np"),
    "238Pu": Isotope("238Pu"),
    "239Pu": Isotope("239Pu"),
    "240Pu": Isotope("240Pu"),
    "241Pu": Isotope("241Pu"),
    "242Pu": Isotope("242Pu"),
    "244Pu": Isotope("244Pu"),
    "241Am": Isotope("241Am"),
    "243Am": Isotope("243Am"),
    "243Cm": Isotope("243Cm"),
    "244Cm": Isotope("244Cm"),
    "245Cm": Isotope("245Cm"),
    "246Cm": Isotope("246Cm"),
    "247Cm": Isotope("247Cm"),
    "248Cm": Isotope("248Cm"),
    "247Bk": Isotope("247Bk"),
    "249Bk": Isotope("249Bk"),
    "249Cf": Isotope("249Cf"),
    "250Cf": Isotope("250Cf"),
    "251Cf": Isotope("251Cf"),
    "252Cf": Isotope("252Cf"),
    "252Es": Isotope("252Es"),
    "257Fm": Isotope("257Fm"),
    "H.": Ion("H."),
    "H1-": Ion("H1-"),
    "H-": None,
    "Li1+": Ion("Li1+"),
    "Li+": None,
    "Be2+": Ion("Be2+"),
    "C.": Ion("C."),
    "O1-": Ion("O1-"),
    "O-": None,
    "O2-": Ion("O2-"),
    "F1-": Ion("F1-"),
    "F-": None,
    "Na1+": Ion("Na1+"),
    "Na+": None,
    "Mg2+": Ion("Mg2+"),
    "Al3+": Ion("Al3+"),
    "Si.": Ion("Si."),
    "Si4+": Ion("Si4+"),
    "Cl1-": Ion("Cl1-"),
    "Cl-": None,
    "K1+": Ion("K1+"),
    "K+": None,
    "Ca2+": Ion("Ca2+"),
    "Sc3+": Ion("Sc3+"),
    "Ti2+": Ion("Ti2+"),
    "Ti3+": Ion("Ti3+"),
    "Ti4+": Ion("Ti4+"),
    "V2+": Ion("V2+"),
    "V3+": Ion("V3+"),
    "V5+": Ion("V5+"),
    "Cr2+": Ion("Cr2+"),
    "Cr3+": Ion("Cr3+"),
    "Mn2+": Ion("Mn2+"),
    "Mn3+": Ion("Mn3+"),
    "Mn4+": Ion("Mn4+"),
    "Fe2+": Ion("Fe2+"),
    "Fe3+": Ion("Fe3+"),
    "Co2+": Ion("Co2+"),
    "Co3+": Ion("Co3+"),
    "Ni2+": Ion("Ni2+"),
    "Ni3+": Ion("Ni3+"),
    "Cu1+": Ion("Cu1+"),
    "Cu+": None,
    "Cu2+": Ion("Cu2+"),
    "Zn2+": Ion("Zn2+"),
    "Ga3+": Ion("Ga3+"),
    "Ge4+": Ion("Ge4+"),
    "Br1-": Ion("Br1-"),
    "Br-": None,
    "Rb1+": Ion("Rb1+"),
    "Rb+": None,
    "Sr2+": Ion("Sr2+"),
    "Y3+": Ion("Y3+"),
    "Zr4+": Ion("Zr4+"),
    "Nb3+": Ion("Nb3+"),
    "Nb5+": Ion("Nb5+"),
    "Mo3+": Ion("Mo3+"),
    "Mo5+": Ion("Mo5+"),
    "Mo6+": Ion("Mo6+"),
    "Ru3+": Ion("Ru3+"),
    "Ru4+": Ion("Ru4+"),
    "Rh3+": Ion("Rh3+"),
    "Rh4+": Ion("Rh4+"),
    "Pd2+": Ion("Pd2+"),
    "Pd4+": Ion("Pd4+"),
    "Ag1+": Ion("Ag1+"),
    "Ag+": None,
    "Ag2+": Ion("Ag2+"),
    "Cd2+": Ion("Cd2+"),
    "In3+": Ion("In3+"),
    "Sn2+": Ion("Sn2+"),
    "Sn4+": Ion("Sn4+"),
    "Sb3+": Ion("Sb3+"),
    "Sb5+": Ion("Sb5+"),
    "I1-": Ion("I1-"),
    "I-": None,
    "Cs1+": Ion("Cs1+"),
    "Cs+": None,
    "Ba2+": Ion("Ba2+"),
    "La3+": Ion("La3+"),
    "Ce3+": Ion("Ce3+"),
    "Ce4+": Ion("Ce4+"),
    "Pr3+": Ion("Pr3+"),
    "Pr4+": Ion("Pr4+"),
    "Nd3+": Ion("Nd3+"),
    "Pm3+": Ion("Pm3+"),
    "Sm3+": Ion("Sm3+"),
    "Eu2+": Ion("Eu2+"),
    "Eu3+": Ion("Eu3+"),
    "Gd3+": Ion("Gd3+"),
    "Tb3+": Ion("Tb3+"),
    "Dy3+": Ion("Dy3+"),
    "Ho3+": Ion("Ho3+"),
    "Er3+": Ion("Er3+"),
    "Tm3+": Ion("Tm3+"),
    "Yb2+": Ion("Yb2+"),
    "Yb3+": Ion("Yb3+"),
    "Lu3+": Ion("Lu3+"),
    "Hf4+": Ion("Hf4+"),
    "Ta5+": Ion("Ta5+"),
    "W6+": Ion("W6+"),
    "Os4+": Ion("Os4+"),
    "Ir3+": Ion("Ir3+"),
    "Ir4+": Ion("Ir4+"),
    "Pt2+": Ion("Pt2+"),
    "Pt4+": Ion("Pt4+"),
    "Au1+": Ion("Au1+"),
    "Au+": None,
    "Au3+": Ion("Au3+"),
    "Hg1+": Ion("Hg1+"),
    "Hg+": None,
    "Hg2+": Ion("Hg2+"),
    "Tl1+": Ion("Tl1+"),
    "Tl+": None,
    "Tl3+": Ion("Tl3+"),
    "Pb2+": Ion("Pb2+"),
    "Pb4+": Ion("Pb4+"),
    "Bi3+": Ion("Bi3+"),
    "Bi5+": Ion("Bi5+"),
    "Ra2+": Ion("Ra2+"),
    "Ac3+": Ion("Ac3+"),
    "Th4+": Ion("Th4+"),
    "U3+": Ion("U3+"),
    "U4+": Ion("U4+"),
    "U6+": Ion("U6+"),
    "Np3+": Ion("Np3+"),
    "Np4+": Ion("Np4+"),
    "Np6+": Ion("Np6+"),
    "Pu3+": Ion("Pu3+"),
    "Pu4+": Ion("Pu4+"),
    "Pu6+": Ion("Pu6+"),
}

Elements.update({'Li+': Elements['Li1+'],
                 "Na+": Elements['Na1+'],
                 "K+": Elements['K1+'],
                 "Cu+": Elements['Cu1+'],
                 "Rb+": Elements['Rb1+'],
                 "Ag+": Elements['Ag1+'],
                 "Cs+": Elements['Cs1+'],
                 "Au+": Elements['Au1+'],
                 "Hg+": Elements['Hg1+'],
                 "Tl+": Elements['Tl1+'],
                 "H-": Elements['H1-'],
                 "O-": Elements['O1-'],
                 "F-": Elements['F1-'],
                 "Cl-": Elements['Cl1-'],
                 "Br-": Elements['Br1-'],
                 "I-": Elements['I1-'],

                 })

ElementsHill = dict(zip(Elements.keys(), [0] * len(Elements)))

H = Elements['H']
He = Elements['He']
Li = Elements['Li']
Be = Elements['Be']
B = Elements['B']
C = Elements['C']
N = Elements['N']
O = Elements['O']
F = Elements['F']
Ne = Elements['Ne']
Na = Elements['Na']
Mg = Elements['Mg']
Al = Elements['Al']
Si = Elements['Si']
P = Elements['P']
S = Elements['S']
Cl = Elements['Cl']
Ar = Elements['Ar']
K = Elements['K']
Ca = Elements['Ca']
Sc = Elements['Sc']
Ti = Elements['Ti']
V = Elements['V']
Cr = Elements['Cr']
Mn = Elements['Mn']
Fe = Elements['Fe']
Co = Elements['Co']
Ni = Elements['Ni']
Cu = Elements['Cu']
Zn = Elements['Zn']
Ga = Elements['Ga']
Ge = Elements['Ge']
As = Elements['As']
Se = Elements['Se']
Br = Elements['Br']
Kr = Elements['Kr']
Rb = Elements['Rb']
Sr = Elements['Sr']
Y = Elements['Y']
Zr = Elements['Zr']
Nb = Elements['Nb']
Mo = Elements['Mo']
Tc = Elements['Tc']
Ru = Elements['Ru']
Rh = Elements['Rh']
Pd = Elements['Pd']
Ag = Elements['Ag']
Cd = Elements['Cd']
In = Elements['In']
Sn = Elements['Sn']
Sb = Elements['Sb']
Te = Elements['Te']
I = Elements['I']
Xe = Elements['Xe']
Cs = Elements['Cs']
Ba = Elements['Ba']
La = Elements['La']
Ce = Elements['Ce']
Pr = Elements['Pr']
Nd = Elements['Nd']
Pm = Elements['Pm']
Sm = Elements['Sm']
Eu = Elements['Eu']
Gd = Elements['Gd']
Tb = Elements['Tb']
Dy = Elements['Dy']
Ho = Elements['Ho']
Er = Elements['Er']
Tm = Elements['Tm']
Yb = Elements['Yb']
Lu = Elements['Lu']
Hf = Elements['Hf']
Ta = Elements['Ta']
W = Elements['W']
Re = Elements['Re']
Os = Elements['Os']
Ir = Elements['Ir']
Pt = Elements['Pt']
Au = Elements['Au']
Hg = Elements['Hg']
Tl = Elements['Tl']
Pb = Elements['Pb']
Bi = Elements['Bi']
Po = Elements['Po']
At = Elements['At']
Rn = Elements['Rn']
Fr = Elements['Fr']
Ra = Elements['Ra']
Ac = Elements['Ac']
Th = Elements['Th']
Pa = Elements['Pa']
U = Elements['U']
Np = Elements['Np']
Pu = Elements['Pu']
Am = Elements['Am']
Cm = Elements['Cm']
Bk = Elements['Bk']
Cf = Elements['Cf']
Es = Elements['Es']
Fm = Elements['Fm']
Md = Elements['Md']
No = Elements['No']
Lr = Elements['Lr']
Rf = Elements['Rf']
Db = Elements['Db']
Sg = Elements['Sg']
Bh = Elements['Bh']
Hs = Elements['Hs']
Mt = Elements['Mt']
Ds = Elements['Ds']
Rg = Elements['Rg']
Cn = Elements['Cn']
Nh = Elements['Nh']
Fl = Elements['Fl']
Mc = Elements['Mc']
Lv = Elements['Lv']
Ts = Elements['Ts']
Og = Elements['Og']
