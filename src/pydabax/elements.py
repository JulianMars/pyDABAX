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


def enforce_unit(unit):
    def decorator(function):
        def wrapper(*args, **kwargs):
            if len(args) == 2:
                args = list(args)
                args[1] = Quantity(args[1], unit())

            result = function(*args, **kwargs)
            if result is not None:
                return Quantity(result, unit())

        return wrapper

    return decorator


class UnitSettings:
    SHOW_UNIT = True
    UNIT_Q = "Å^-1"
    UNIT_E = "keV"
    UNIT_TTH = "°"
    UNIT_R = "Å"


class BaseElement:
    SHOW_UNIT = True
    UNIT_Q = "Å^-1"
    UNIT_E = "keV"
    UNIT_TTH = "°"
    UNIT_R = "Å"

    def __init__(self, q="0 1/Å", photon_energy="8.047 keV"):
        self._q = None
        self._energy = None
        self._l = None
        self._ttheta = None
        self.energy = photon_energy
        self.q = q

    @property
    @show_unit
    @enforce_unit((lambda: UnitSettings.UNIT_Q))
    def q(self):
        ans = self._q
        return ans

    @q.setter
    @enforce_unit((lambda: UnitSettings.UNIT_Q))
    def q(self, value):
        self._q = value

    @property
    @show_unit
    @enforce_unit((lambda: UnitSettings.UNIT_TTH))
    def ttheta(self):
        ans = Element.calc_ttheta(self.energy, self.q)
        return ans

    @ttheta.setter
    @enforce_unit((lambda: UnitSettings.UNIT_TTH))
    def ttheta(self, value):
        self._q = Element.calc_q(self.energy, value)

    @property
    @show_unit
    @enforce_unit((lambda: UnitSettings.UNIT_E))
    def energy(self):
        ans = self._energy
        return ans

    @energy.setter
    @enforce_unit((lambda: UnitSettings.UNIT_E))
    def energy(self, value):
        self._energy = value

    @staticmethod
    def calc_dabax(q, df):
        def f0():
            a, b, const = params
            summ = 0
            for _a, _b in zip(a, b):
                summ += _a * np.exp(-_b * k ** 2)
            summ += const
            return summ

        params = [df.values[0, :5],  # a
                  df.values[0, 6:],  # b
                  df.values[0, 5]]  # c

        k = Quantity(q / (4 * np.pi), "1/Å").value  # Do not change; TABLES IN Å-1
        res = f0()
        return res

    @staticmethod
    def interpolate_chantler(energy, df):
        nist = df.values.astype(float).T
        arr = np.abs(nist[0] - energy)
        idx = arr.argsort()[:2]  # find nearest data-points
        out = []

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
        energy = Quantity(energy, BaseElement.UNIT_E)
        wavelength = Quantity(hc / energy, BaseElement.UNIT_R)
        ttheta = Quantity(ttheta, BaseElement.UNIT_TTH)
        q = 4 * np.pi / wavelength * np.sin(ttheta / 2)
        return q

    @staticmethod
    def calc_ttheta(energy, q):
        energy = Quantity(energy, BaseElement.UNIT_E)
        wavelength = Quantity(hc / energy, BaseElement.UNIT_R)
        q = Quantity(q, BaseElement.UNIT_Q)
        wavelength = Quantity(wavelength, BaseElement.UNIT_R)
        res = 2 * np.arcsin(q * wavelength / (4 * np.pi))
        return Quantity(res, BaseElement.UNIT_TTH)

    @staticmethod
    def crossec_compton_kleinnishina(energy, ttheta):
        energy = Quantity(energy, BaseElement.UNIT_E)
        ttheta = Quantity(ttheta, BaseElement.UNIT_TTH)
        d = 1 + (energy / (c.m_e * c.c ** 2)) * (
                1 - np.cos(ttheta))
        p = 1 / d  # P Klein-Nishina Equation
        diff_crossec = 1 / 2 * p ** 2 * (p + 1 / p - np.sin(ttheta) ** 2)
        return diff_crossec, energy * p

    @property
    @show_unit
    def wavelength(self):
        ans = Quantity(hc / self._energy, BaseElement.UNIT_R)
        return ans  # if Element.SHOW_UNIT else ans.value

    @wavelength.setter
    def wavelength(self, value):
        self._energy = Quantity(hc / value, BaseElement.UNIT_E)


class Element(BaseElement):
    def __init__(self, symbol, q="0", energy="8.047 keV"):
        super().__init__(q, energy)
        self.symbol = symbol

    def _repr_html_(self):

        try:
            aw = self.atomic_weight
        except:
            aw = '-'
        try:
            ed = self.edges._repr_html_()
        except:
            ed = ''
        ans = ("<h1>{}</h1>".format(self.name) +
               "<table> <tr>     <tr>    <td>atomic number</td>    <td>{}</td>  </tr><tr>    <td>atomic weight</td>    <td>{}</td>  </tr></table>".format(
                   self.atomic_number, aw) +
               ed)
        return ans

    @property
    def atomic_number(self):
        val = dabax.get_entry(self.symbol, "atomic_number")
        return val

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
    @show_unit
    def molecular_weight(self):
        ans = Quantity(
            dabax.get_entry(self.symbol, "atomic_weight"), u.gram / u.mol
        )
        return ans

    @property
    def atomic_weight(self):
        return self.molecular_weight

    @property
    def f(self):
        ans = self.get_f(self.energy, self.q)
        return ans

    @property
    def crossec_compton(self):
        return self.get_compton(self.energy, self.q)

    @property
    def crossec_thomson(self):
        return self.get_thomson(self.energy, self.q)

    @property
    def edges(self):
        return self._get_nist_edges_chantler()

    @property
    @show_unit
    def k_alpha_2(self):
        val = Quantity(float(self.edges.loc["K"] - self.edges.loc["L II"]), 'keV')
        val = Quantity(val, BaseElement.UNIT_E)
        return val

    @property
    @show_unit
    def k_alpha_1(self):
        val = Quantity(float(self.edges.loc["K"] - self.edges.loc["L III"]), 'keV')
        val = Quantity(val, BaseElement.UNIT_E)
        return val

    @property
    def k_alpha(self):
        return (2 * self.k_alpha_1 + self.k_alpha_1) / 3

    @property
    @show_unit
    def k_beta(self):
        val = Quantity(float(self.edges.loc["K"] - self.edges.loc["M II"]), "keV")
        val = Quantity(val, BaseElement.UNIT_E)
        return val

    @property
    def name(self):
        val = dabax.get_entry(self.symbol, "name")
        return val

    @property
    def neturon_bcoh(self):
        coh, inc, ab = self._get_nist_b_sears()
        return coh

    @property
    def neutron_binc(self):
        coh, inc, ab = self._get_nist_b_sears()
        return inc

    @property
    def neutron_abs_at_2200mps(self):
        coh, inc, ab = self._get_nist_b_sears()
        return ab

    @property
    def mup(self):
        return self.get_mup(self.energy)

    def get_mup(self, energy):
        return self._get_nist_mup_chantler(energy)

    def _get_nist_mup_chantler(self, energy):
        mu = self._get_nist_f1f2mu_chantler(energy)['µ/p total (cm2/g)']
        return Quantity(float(mu), 'cm^2/g')

    def _get_dabax_f0_waaskirf(self, q):
        df = dabax.get_table(self.symbol, "dabax_f0_waaskirf")
        return Element.calc_dabax(q, df)

    def _get_dabax_isf_balyuzi(self, q):
        df = dabax.get_table(self.symbol, "dabax_isf_balyuzi")
        return Element.calc_dabax(q, df)

    def get_f0(self, q):
        return self._get_dabax_f0_waaskirf(q)

    def get_isf(self, q):
        return self._get_dabax_isf_balyuzi(q)

    def get_compton(self, energy, q):
        q = Quantity(q, "1/Å")
        energy = Quantity(energy, "keV")

        ttheta = Element.calc_ttheta(energy, q)

        crossec, energy_out = Element.crossec_compton_kleinnishina(energy, ttheta)
        isf = self.get_isf(q)

        return crossec * (self.atomic_number - isf), energy_out

    def get_thomson(self, energy, q):
        q = Quantity(q, "1/Å")
        energy = Quantity(energy, "keV")

        ttheta = Element.calc_ttheta(energy, q)

        ans = 0.5 * (1 + np.cos(ttheta) ** 2) * abs(self.get_f(energy, q)) ** 2

        return ans

    def _get_nist_f1f2_chantler(self, energy):

        df = dabax.get_table(self.symbol, "nist_f1f2_chantler")

        out = Element.interpolate_chantler(energy, df)

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

        f1 = out[0]
        f2 = out[1]
        rel_corr = float(out[2][1])  # 3/5CL
        nt_corr = float(out[3])
        return [f1 + rel_corr + nt_corr, f2]

    def _get_nist_f1f2mu_chantler(self, energy):
        """
        dabax:f1f2_Chantler
        """
        df = dabax.get_table(self.symbol, "nist_f1f2_chantler")
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
        energy *= 1e3
        df = dabax.get_table(self.symbol, "cxro_f1f2_henke")
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
        return p1(energy), p2(energy)

    def _get_nist_b_sears(self):
        df = dabax.get_table(self.symbol, "nist_b_sears")
        unit = u.barn if Element.SHOW_UNIT else 1
        return (
            float(df["Coh xs (barn)"]) * unit,
            float(df["Inc xs (barn)"]) * unit,
            float(df["Abs xs at 2200m/s (barn)"]) * unit,
        )

    def _get_f1f2(self, energy, databank="auto"):
        if databank == "auto":
            databank = "cxro" if energy <= 30 else "nist"

        if databank in ["CXRO", "cxro", "henke", "Henke"]:
            return self._get_cxro_f1f2_henke(energy)
        if databank in ["NIST", "nist", "chantler", "Chantler"]:
            return self._get_nist_f1f2_chantler(energy)

    def get_f(self, energy, q, **params):
        q = Quantity(q, BaseElement.UNIT_Q)
        energy = Quantity(energy, BaseElement.UNIT_E)

        q = Quantity(q, "1/Å").value  # Ensure parameters are in right unit for table
        energy = Quantity(energy, "keV").value

        f1, f2 = self._get_f1f2(energy, **params)
        f0 = self._get_dabax_f0_waaskirf(q)
        return f0 + f1 - self.atomic_number + f2 * 1j

    def _get_nist_edges_chantler(self):
        df = dabax.get_table(self.symbol, "nist_edges_chantler")
        return df


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
    def __init__(self, formula, energy='8.047 keV', q='0 1/Å', density='1 g/cm^3'):
        super().__init__(q, energy)
        self.formula = formula
        self.composition = formula
        self.density = density

    def __repr__(self):
        return str(self.formula) + '\n' + self.molecular_weight.__str__() + '\n' + str(self.composition)

    def _repr_latex_(self) -> str:

        return r"$\mathrm{{{}}}$".format(self.formula) + \
               r"$\\ \mathrm{m_w}:\,$" + self.molecular_weight._repr_latex_() + \
               r"$\\ \rho:\,$" + self.density._repr_latex_()

    @property
    def composition(self):
        return self._composition

    @composition.setter
    def composition(self, formula):
        self._composition = Compound.parse_formula(formula).get_elements()

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, value):
        self._density = Quantity(value, 'g/cm^3')

    @property
    def f(self):
        res = np.zeros_like(self.q.value, dtype=np.complex128)
        for k in self.composition:
            res += Elements[k].get_f(self.energy, self.q) * self.composition[k]
        return res

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
            mw += Elements[k].atomic_weight * self.composition[k]
        return Quantity(mw, 'g/mol')

    @property
    def q_crit_sq(self):
        qcsq = 16 * np.pi * r_e * self.f / self.molecular_volume

        return Quantity(qcsq, '1/Å^2')

    @property
    def q_crit(self):
        return abs(self.q_crit_sq**0.5)

    @property
    def deltabeta(self):

        db = self.wavelength**2/(2*np.pi)*r_e/self.molecular_volume*self.f
        return db

    @property
    def molecular_volume(self):
        mv = self.molecular_weight / self.density / c.N_A
        return Quantity(mv, 'Å^3')

    @staticmethod
    def parse_string_to_list(element_str):
        res = []
        match = re.findall(
            r"[A-Z][a-z]?\d?[+|-]\d*(?:\.\d+)?|[A-Z][a-z]?\d*(?:\.\d+)?|\((?:[^()]*(?:\(.*\))?[^()]*)+\)\d+(?:\.\d+)?",
            element_str,
        )
        if match == [element_str]:
            submatch = re.match(r"([A-Z][a-z]?(?:\d?[+|-])?)(\d*(?:\.\d+)?)", element_str)
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
                print(element_str)
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
            if isinstance(subunit[1], int):
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
        return counter


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
    "1H": Element("1H"),
    "2H": Element("2H"),
    "3H": Element("3H"),
    "3He": Element("3He"),
    "4He": Element("4He"),
    "6Li": Element("6Li"),
    "7Li": Element("7Li"),
    "10B": Element("10B"),
    "11B": Element("11B"),
    "12C": Element("12C"),
    "13C": Element("13C"),
    "14N": Element("14N"),
    "15N": Element("15N"),
    "16O": Element("16O"),
    "17O": Element("17O"),
    "18O": Element("18O"),
    "20Ne": Element("20Ne"),
    "21Ne": Element("21Ne"),
    "22Ne": Element("22Ne"),
    "24Mg": Element("24Mg"),
    "25Mg": Element("25Mg"),
    "26Mg": Element("26Mg"),
    "28Si": Element("28Si"),
    "29Si": Element("29Si"),
    "30Si": Element("30Si"),
    "32S": Element("32S"),
    "33S": Element("33S"),
    "34S": Element("34S"),
    "36S": Element("36S"),
    "35Cl": Element("35Cl"),
    "37Cl": Element("37Cl"),
    "36Ar": Element("36Ar"),
    "38Ar": Element("38Ar"),
    "40Ar": Element("40Ar"),
    "39K": Element("39K"),
    "40K": Element("40K"),
    "41K": Element("41K"),
    "40Ca": Element("40Ca"),
    "42Ca": Element("42Ca"),
    "43Ca": Element("43Ca"),
    "44Ca": Element("44Ca"),
    "46Ca": Element("46Ca"),
    "48Ca": Element("48Ca"),
    "46Ti": Element("46Ti"),
    "47Ti": Element("47Ti"),
    "48Ti": Element("48Ti"),
    "49Ti": Element("49Ti"),
    "50Ti": Element("50Ti"),
    "50V": Element("50V"),
    "51V": Element("51V"),
    "50Cr": Element("50Cr"),
    "52Cr": Element("52Cr"),
    "53Cr": Element("53Cr"),
    "54Cr": Element("54Cr"),
    "54Fe": Element("54Fe"),
    "56Fe": Element("56Fe"),
    "57Fe": Element("57Fe"),
    "58Fe": Element("58Fe"),
    "58Ni": Element("58Ni"),
    "60Ni": Element("60Ni"),
    "61Ni": Element("61Ni"),
    "62Ni": Element("62Ni"),
    "64Ni": Element("64Ni"),
    "63Cu": Element("63Cu"),
    "65Cu": Element("65Cu"),
    "64Zn": Element("64Zn"),
    "66Zn": Element("66Zn"),
    "67Zn": Element("67Zn"),
    "68Zn": Element("68Zn"),
    "70Zn": Element("70Zn"),
    "69Ga": Element("69Ga"),
    "71Ga": Element("71Ga"),
    "70Ge": Element("70Ge"),
    "72Ge": Element("72Ge"),
    "73Ge": Element("73Ge"),
    "74Ge": Element("74Ge"),
    "76Ge": Element("76Ge"),
    "74Se": Element("74Se"),
    "76Se": Element("76Se"),
    "77Se": Element("77Se"),
    "78Se": Element("78Se"),
    "80Se": Element("80Se"),
    "82Se": Element("82Se"),
    "79Br": Element("79Br"),
    "81Br": Element("81Br"),
    "78Kr": Element("78Kr"),
    "80Kr": Element("80Kr"),
    "82Kr": Element("82Kr"),
    "83Kr": Element("83Kr"),
    "84Kr": Element("84Kr"),
    "86Kr": Element("86Kr"),
    "85Rb": Element("85Rb"),
    "87Rb": Element("87Rb"),
    "84Sr": Element("84Sr"),
    "86Sr": Element("86Sr"),
    "87Sr": Element("87Sr"),
    "88Sr": Element("88Sr"),
    "90Zr": Element("90Zr"),
    "91Zr": Element("91Zr"),
    "92Zr": Element("92Zr"),
    "94Zr": Element("94Zr"),
    "96Zr": Element("96Zr"),
    "92Mo": Element("92Mo"),
    "94Mo": Element("94Mo"),
    "95Mo": Element("95Mo"),
    "96Mo": Element("96Mo"),
    "97Mo": Element("97Mo"),
    "98Mo": Element("98Mo"),
    "100Mo": Element("100Mo"),
    "96Ru": Element("96Ru"),
    "98Ru": Element("98Ru"),
    "99Ru": Element("99Ru"),
    "100Ru": Element("100Ru"),
    "101Ru": Element("101Ru"),
    "102Ru": Element("102Ru"),
    "104Ru": Element("104Ru"),
    "102Pd": Element("102Pd"),
    "104Pd": Element("104Pd"),
    "105Pd": Element("105Pd"),
    "106Pd": Element("106Pd"),
    "108Pd": Element("108Pd"),
    "110Pd": Element("110Pd"),
    "107Ag": Element("107Ag"),
    "109Ag": Element("109Ag"),
    "106Cd": Element("106Cd"),
    "108Cd": Element("108Cd"),
    "110Cd": Element("110Cd"),
    "111Cd": Element("111Cd"),
    "112Cd": Element("112Cd"),
    "113Cd": Element("113Cd"),
    "114Cd": Element("114Cd"),
    "116Cd": Element("116Cd"),
    "113In": Element("113In"),
    "115In": Element("115In"),
    "112Sn": Element("112Sn"),
    "114Sn": Element("114Sn"),
    "115Sn": Element("115Sn"),
    "116Sn": Element("116Sn"),
    "117Sn": Element("117Sn"),
    "118Sn": Element("118Sn"),
    "119Sn": Element("119Sn"),
    "120Sn": Element("120Sn"),
    "122Sn": Element("122Sn"),
    "124Sn": Element("124Sn"),
    "121Sb": Element("121Sb"),
    "123Sb": Element("123Sb"),
    "120Te": Element("120Te"),
    "122Te": Element("122Te"),
    "123Te": Element("123Te"),
    "124Te": Element("124Te"),
    "125Te": Element("125Te"),
    "126Te": Element("126Te"),
    "128Te": Element("128Te"),
    "130Te": Element("130Te"),
    "124Xe": Element("124Xe"),
    "126Xe": Element("126Xe"),
    "128Xe": Element("128Xe"),
    "129Xe": Element("129Xe"),
    "130Xe": Element("130Xe"),
    "131Xe": Element("131Xe"),
    "132Xe": Element("132Xe"),
    "134Xe": Element("134Xe"),
    "136Xe": Element("136Xe"),
    "130Ba": Element("130Ba"),
    "132Ba": Element("132Ba"),
    "134Ba": Element("134Ba"),
    "135Ba": Element("135Ba"),
    "136Ba": Element("136Ba"),
    "137Ba": Element("137Ba"),
    "138Ba": Element("138Ba"),
    "138La": Element("138La"),
    "139La": Element("139La"),
    "136Ce": Element("136Ce"),
    "138Ce": Element("138Ce"),
    "140Ce": Element("140Ce"),
    "142Ce": Element("142Ce"),
    "142Nd": Element("142Nd"),
    "143Nd": Element("143Nd"),
    "144Nd": Element("144Nd"),
    "145Nd": Element("145Nd"),
    "146Nd": Element("146Nd"),
    "148Nd": Element("148Nd"),
    "150Nd": Element("150Nd"),
    "144Sm": Element("144Sm"),
    "147Sm": Element("147Sm"),
    "148Sm": Element("148Sm"),
    "149Sm": Element("149Sm"),
    "150Sm": Element("150Sm"),
    "152Sm": Element("152Sm"),
    "154Sm": Element("154Sm"),
    "151Eu": Element("151Eu"),
    "153Eu": Element("153Eu"),
    "152Gd": Element("152Gd"),
    "154Gd": Element("154Gd"),
    "155Gd": Element("155Gd"),
    "156Gd": Element("156Gd"),
    "157Gd": Element("157Gd"),
    "158Gd": Element("158Gd"),
    "160Gd": Element("160Gd"),
    "156Dy": Element("156Dy"),
    "158Dy": Element("158Dy"),
    "160Dy": Element("160Dy"),
    "161Dy": Element("161Dy"),
    "162Dy": Element("162Dy"),
    "163Dy": Element("163Dy"),
    "164Dy": Element("164Dy"),
    "162Er": Element("162Er"),
    "164Er": Element("164Er"),
    "166Er": Element("166Er"),
    "167Er": Element("167Er"),
    "168Er": Element("168Er"),
    "170Er": Element("170Er"),
    "168Yb": Element("168Yb"),
    "170Yb": Element("170Yb"),
    "171Yb": Element("171Yb"),
    "172Yb": Element("172Yb"),
    "173Yb": Element("173Yb"),
    "174Yb": Element("174Yb"),
    "176Yb": Element("176Yb"),
    "175Lu": Element("175Lu"),
    "176Lu": Element("176Lu"),
    "174Hf": Element("174Hf"),
    "176Hf": Element("176Hf"),
    "177Hf": Element("177Hf"),
    "178Hf": Element("178Hf"),
    "179Hf": Element("179Hf"),
    "180Hf": Element("180Hf"),
    "180Ta": Element("180Ta"),
    "181Ta": Element("181Ta"),
    "180W": Element("180W"),
    "182W": Element("182W"),
    "183W": Element("183W"),
    "184W": Element("184W"),
    "186W": Element("186W"),
    "185Re": Element("185Re"),
    "187Re": Element("187Re"),
    "184Os": Element("184Os"),
    "186Os": Element("186Os"),
    "187Os": Element("187Os"),
    "188Os": Element("188Os"),
    "189Os": Element("189Os"),
    "190Os": Element("190Os"),
    "192Os": Element("192Os"),
    "191Ir": Element("191Ir"),
    "193Ir": Element("193Ir"),
    "190Pt": Element("190Pt"),
    "192Pt": Element("192Pt"),
    "194Pt": Element("194Pt"),
    "195Pt": Element("195Pt"),
    "196Pt": Element("196Pt"),
    "198Pt": Element("198Pt"),
    "196Hg": Element("196Hg"),
    "198Hg": Element("198Hg"),
    "199Hg": Element("199Hg"),
    "200Hg": Element("200Hg"),
    "201Hg": Element("201Hg"),
    "202Hg": Element("202Hg"),
    "204Hg": Element("204Hg"),
    "203Tl": Element("203Tl"),
    "205Tl": Element("205Tl"),
    "204Pb": Element("204Pb"),
    "206Pb": Element("206Pb"),
    "207Pb": Element("207Pb"),
    "208Pb": Element("208Pb"),
    "233U": Element("233U"),
    "234U": Element("234U"),
    "235U": Element("235U"),
    "238U": Element("238U"),
    "238Pu": Element("238Pu"),
    "239Pu": Element("239Pu"),
    "240Pu": Element("240Pu"),
    "242Pu": Element("242Pu"),
    "244Cm": Element("244Cm"),
    "246Cm": Element("246Cm"),
    "248Cm": Element("248Cm"),
    "H1-": Element("H1-"),
    "Li1+": Element("Li1+"),
    "Be2+": Element("Be2+"),
    "Cval": Element("Cval"),
    "O1-": Element("O1-"),
    "O2-": Element("O2-"),
    "F1-": Element("F1-"),
    "Na1+": Element("Na1+"),
    "Mg2+": Element("Mg2+"),
    "Al3+": Element("Al3+"),
    "Siva": Element("Siva"),
    "Si4+": Element("Si4+"),
    "Cl1-": Element("Cl1-"),
    "K1+": Element("K1+"),
    "Ca2+": Element("Ca2+"),
    "Sc3+": Element("Sc3+"),
    "Ti2+": Element("Ti2+"),
    "Ti3+": Element("Ti3+"),
    "Ti4+": Element("Ti4+"),
    "V2+": Element("V2+"),
    "V3+": Element("V3+"),
    "V5+": Element("V5+"),
    "Cr2+": Element("Cr2+"),
    "Cr3+": Element("Cr3+"),
    "Mn2+": Element("Mn2+"),
    "Mn3+": Element("Mn3+"),
    "Mn4+": Element("Mn4+"),
    "Fe2+": Element("Fe2+"),
    "Fe3+": Element("Fe3+"),
    "Co2+": Element("Co2+"),
    "Co3+": Element("Co3+"),
    "Ni2+": Element("Ni2+"),
    "Ni3+": Element("Ni3+"),
    "Cu1+": Element("Cu1+"),
    "Cu2+": Element("Cu2+"),
    "Zn2+": Element("Zn2+"),
    "Ga3+": Element("Ga3+"),
    "Ge4+": Element("Ge4+"),
    "Br1-": Element("Br1-"),
    "Rb1+": Element("Rb1+"),
    "Sr2+": Element("Sr2+"),
    "Zr4+": Element("Zr4+"),
    "Nb3+": Element("Nb3+"),
    "Nb5+": Element("Nb5+"),
    "Mo3+": Element("Mo3+"),
    "Mo5+": Element("Mo5+"),
    "Mo6+": Element("Mo6+"),
    "Ru3+": Element("Ru3+"),
    "Ru4+": Element("Ru4+"),
    "Rh3+": Element("Rh3+"),
    "Rh4+": Element("Rh4+"),
    "Pd2+": Element("Pd2+"),
    "Pd4+": Element("Pd4+"),
    "Ag1+": Element("Ag1+"),
    "Ag2+": Element("Ag2+"),
    "Cd2+": Element("Cd2+"),
    "In3+": Element("In3+"),
    "Sn2+": Element("Sn2+"),
    "Sn4+": Element("Sn4+"),
    "Sb3+": Element("Sb3+"),
    "Sb5+": Element("Sb5+"),
    "I1-": Element("I1-"),
    "Cs1+": Element("Cs1+"),
    "Ba2+": Element("Ba2+"),
    "La3+": Element("La3+"),
    "Ce3+": Element("Ce3+"),
    "Ce4+": Element("Ce4+"),
    "Pr3+": Element("Pr3+"),
    "Pr4+": Element("Pr4+"),
    "Nd3+": Element("Nd3+"),
    "Pm3+": Element("Pm3+"),
    "Sm3+": Element("Sm3+"),
    "Eu2+": Element("Eu2+"),
    "Eu3+": Element("Eu3+"),
    "Gd3+": Element("Gd3+"),
    "Tb3+": Element("Tb3+"),
    "Dy3+": Element("Dy3+"),
    "Ho3+": Element("Ho3+"),
    "Er3+": Element("Er3+"),
    "Tm3+": Element("Tm3+"),
    "Yb2+": Element("Yb2+"),
    "Yb3+": Element("Yb3+"),
    "Lu3+": Element("Lu3+"),
    "Hf4+": Element("Hf4+"),
    "Ta5+": Element("Ta5+"),
    "W6+": Element("W6+"),
    "Os4+": Element("Os4+"),
    "Ir3+": Element("Ir3+"),
    "Ir4+": Element("Ir4+"),
    "Pt2+": Element("Pt2+"),
    "Pt4+": Element("Pt4+"),
    "Au1+": Element("Au1+"),
    "Au3+": Element("Au3+"),
    "Hg1+": Element("Hg1+"),
    "Hg2+": Element("Hg2+"),
    "Tl1+": Element("Tl1+"),
    "Tl3+": Element("Tl3+"),
    "Pb2+": Element("Pb2+"),
    "Pb4+": Element("Pb4+"),
    "Bi3+": Element("Bi3+"),
    "Bi5+": Element("Bi5+"),
    "Ra2+": Element("Ra2+"),
    "Ac3+": Element("Ac3+"),
    "Th4+": Element("Th4+"),
    "U3+": Element("U3+"),
    "U4+": Element("U4+"),
    "U6+": Element("U6+"),
    "Np3+": Element("Np3+"),
    "Np4+": Element("Np4+"),
    "Np6+": Element("Np6+"),
    "Pu3+": Element("Pu3+"),
    "Pu4+": Element("Pu4+"),
    "Pu6+": Element("Pu6+"),
}

ElementsHill = {
    "C": 0,
    "H": 0,
    "Ac": 0,
    "Ag": 0,
    "Al": 0,
    "Am": 0,
    "Ar": 0,
    "As": 0,
    "At": 0,
    "Au": 0,
    "B": 0,
    "Ba": 0,
    "Be": 0,
    "Bh": 0,
    "Bi": 0,
    "Bk": 0,
    "Br": 0,
    "Ca": 0,
    "Cd": 0,
    "Ce": 0,
    "Cf": 0,
    "Cl": 0,
    "Cm": 0,
    "Cn": 0,
    "Co": 0,
    "Cr": 0,
    "Cs": 0,
    "Cu": 0,
    "Db": 0,
    "Ds": 0,
    "Dy": 0,
    "Er": 0,
    "Es": 0,
    "Eu": 0,
    "F": 0,
    "Fe": 0,
    "Fl": 0,
    "Fm": 0,
    "Fr": 0,
    "Ga": 0,
    "Gd": 0,
    "Ge": 0,
    "He": 0,
    "Hf": 0,
    "Hg": 0,
    "Ho": 0,
    "Hs": 0,
    "I": 0,
    "In": 0,
    "Ir": 0,
    "K": 0,
    "Kr": 0,
    "La": 0,
    "Li": 0,
    "Lr": 0,
    "Lu": 0,
    "Lv": 0,
    "Mc": 0,
    "Md": 0,
    "Mg": 0,
    "Mn": 0,
    "Mo": 0,
    "Mt": 0,
    "N": 0,
    "Na": 0,
    "Nb": 0,
    "Nd": 0,
    "Ne": 0,
    "Nh": 0,
    "Ni": 0,
    "No": 0,
    "Np": 0,
    "O": 0,
    "Og": 0,
    "Os": 0,
    "P": 0,
    "Pa": 0,
    "Pb": 0,
    "Pd": 0,
    "Pm": 0,
    "Po": 0,
    "Pr": 0,
    "Pt": 0,
    "Pu": 0,
    "Ra": 0,
    "Rb": 0,
    "Re": 0,
    "Rf": 0,
    "Rg": 0,
    "Rh": 0,
    "Rn": 0,
    "Ru": 0,
    "S": 0,
    "Sb": 0,
    "Sc": 0,
    "Se": 0,
    "Sg": 0,
    "Si": 0,
    "Sm": 0,
    "Sn": 0,
    "Sr": 0,
    "Ta": 0,
    "Tb": 0,
    "Tc": 0,
    "Te": 0,
    "Th": 0,
    "Ti": 0,
    "Tl": 0,
    "Tm": 0,
    "Ts": 0,
    "U": 0,
    "V": 0,
    "W": 0,
    "Xe": 0,
    "Y": 0,
    "Yb": 0,
    "Zn": 0,
    "Zr": 0,
    "1H": 0,
    "2H": 0,
    "3H": 0,
    "3He": 0,
    "4He": 0,
    "6Li": 0,
    "7Li": 0,
    "10B": 0,
    "11B": 0,
    "12C": 0,
    "13C": 0,
    "14N": 0,
    "15N": 0,
    "16O": 0,
    "17O": 0,
    "18O": 0,
    "20Ne": 0,
    "21Ne": 0,
    "22Ne": 0,
    "24Mg": 0,
    "25Mg": 0,
    "26Mg": 0,
    "28Si": 0,
    "29Si": 0,
    "30Si": 0,
    "32S": 0,
    "33S": 0,
    "34S": 0,
    "36S": 0,
    "35Cl": 0,
    "37Cl": 0,
    "36Ar": 0,
    "38Ar": 0,
    "40Ar": 0,
    "39K": 0,
    "40K": 0,
    "41K": 0,
    "40Ca": 0,
    "42Ca": 0,
    "43Ca": 0,
    "44Ca": 0,
    "46Ca": 0,
    "48Ca": 0,
    "46Ti": 0,
    "47Ti": 0,
    "48Ti": 0,
    "49Ti": 0,
    "50Ti": 0,
    "50V": 0,
    "51V": 0,
    "50Cr": 0,
    "52Cr": 0,
    "53Cr": 0,
    "54Cr": 0,
    "54Fe": 0,
    "56Fe": 0,
    "57Fe": 0,
    "58Fe": 0,
    "58Ni": 0,
    "60Ni": 0,
    "61Ni": 0,
    "62Ni": 0,
    "64Ni": 0,
    "63Cu": 0,
    "65Cu": 0,
    "64Zn": 0,
    "66Zn": 0,
    "67Zn": 0,
    "68Zn": 0,
    "70Zn": 0,
    "69Ga": 0,
    "71Ga": 0,
    "70Ge": 0,
    "72Ge": 0,
    "73Ge": 0,
    "74Ge": 0,
    "76Ge": 0,
    "74Se": 0,
    "76Se": 0,
    "77Se": 0,
    "78Se": 0,
    "80Se": 0,
    "82Se": 0,
    "79Br": 0,
    "81Br": 0,
    "78Kr": 0,
    "80Kr": 0,
    "82Kr": 0,
    "83Kr": 0,
    "84Kr": 0,
    "86Kr": 0,
    "85Rb": 0,
    "87Rb": 0,
    "84Sr": 0,
    "86Sr": 0,
    "87Sr": 0,
    "88Sr": 0,
    "90Zr": 0,
    "91Zr": 0,
    "92Zr": 0,
    "94Zr": 0,
    "96Zr": 0,
    "92Mo": 0,
    "94Mo": 0,
    "95Mo": 0,
    "96Mo": 0,
    "97Mo": 0,
    "98Mo": 0,
    "100Mo": 0,
    "96Ru": 0,
    "98Ru": 0,
    "99Ru": 0,
    "100Ru": 0,
    "101Ru": 0,
    "102Ru": 0,
    "104Ru": 0,
    "102Pd": 0,
    "104Pd": 0,
    "105Pd": 0,
    "106Pd": 0,
    "108Pd": 0,
    "110Pd": 0,
    "107Ag": 0,
    "109Ag": 0,
    "106Cd": 0,
    "108Cd": 0,
    "110Cd": 0,
    "111Cd": 0,
    "112Cd": 0,
    "113Cd": 0,
    "114Cd": 0,
    "116Cd": 0,
    "113In": 0,
    "115In": 0,
    "112Sn": 0,
    "114Sn": 0,
    "115Sn": 0,
    "116Sn": 0,
    "117Sn": 0,
    "118Sn": 0,
    "119Sn": 0,
    "120Sn": 0,
    "122Sn": 0,
    "124Sn": 0,
    "121Sb": 0,
    "123Sb": 0,
    "120Te": 0,
    "122Te": 0,
    "123Te": 0,
    "124Te": 0,
    "125Te": 0,
    "126Te": 0,
    "128Te": 0,
    "130Te": 0,
    "124Xe": 0,
    "126Xe": 0,
    "128Xe": 0,
    "129Xe": 0,
    "130Xe": 0,
    "131Xe": 0,
    "132Xe": 0,
    "134Xe": 0,
    "136Xe": 0,
    "130Ba": 0,
    "132Ba": 0,
    "134Ba": 0,
    "135Ba": 0,
    "136Ba": 0,
    "137Ba": 0,
    "138Ba": 0,
    "138La": 0,
    "139La": 0,
    "136Ce": 0,
    "138Ce": 0,
    "140Ce": 0,
    "142Ce": 0,
    "142Nd": 0,
    "143Nd": 0,
    "144Nd": 0,
    "145Nd": 0,
    "146Nd": 0,
    "148Nd": 0,
    "150Nd": 0,
    "144Sm": 0,
    "147Sm": 0,
    "148Sm": 0,
    "149Sm": 0,
    "150Sm": 0,
    "152Sm": 0,
    "154Sm": 0,
    "151Eu": 0,
    "153Eu": 0,
    "152Gd": 0,
    "154Gd": 0,
    "155Gd": 0,
    "156Gd": 0,
    "157Gd": 0,
    "158Gd": 0,
    "160Gd": 0,
    "156Dy": 0,
    "158Dy": 0,
    "160Dy": 0,
    "161Dy": 0,
    "162Dy": 0,
    "163Dy": 0,
    "164Dy": 0,
    "162Er": 0,
    "164Er": 0,
    "166Er": 0,
    "167Er": 0,
    "168Er": 0,
    "170Er": 0,
    "168Yb": 0,
    "170Yb": 0,
    "171Yb": 0,
    "172Yb": 0,
    "173Yb": 0,
    "174Yb": 0,
    "176Yb": 0,
    "175Lu": 0,
    "176Lu": 0,
    "174Hf": 0,
    "176Hf": 0,
    "177Hf": 0,
    "178Hf": 0,
    "179Hf": 0,
    "180Hf": 0,
    "180Ta": 0,
    "181Ta": 0,
    "180W": 0,
    "182W": 0,
    "183W": 0,
    "184W": 0,
    "186W": 0,
    "185Re": 0,
    "187Re": 0,
    "184Os": 0,
    "186Os": 0,
    "187Os": 0,
    "188Os": 0,
    "189Os": 0,
    "190Os": 0,
    "192Os": 0,
    "191Ir": 0,
    "193Ir": 0,
    "190Pt": 0,
    "192Pt": 0,
    "194Pt": 0,
    "195Pt": 0,
    "196Pt": 0,
    "198Pt": 0,
    "196Hg": 0,
    "198Hg": 0,
    "199Hg": 0,
    "200Hg": 0,
    "201Hg": 0,
    "202Hg": 0,
    "204Hg": 0,
    "203Tl": 0,
    "205Tl": 0,
    "204Pb": 0,
    "206Pb": 0,
    "207Pb": 0,
    "208Pb": 0,
    "233U": 0,
    "234U": 0,
    "235U": 0,
    "238U": 0,
    "238Pu": 0,
    "239Pu": 0,
    "240Pu": 0,
    "242Pu": 0,
    "244Cm": 0,
    "246Cm": 0,
    "248Cm": 0,
    "H1-": 0,
    "Li1+": 0,
    "Be2+": 0,
    "Cval": 0,
    "O1-": 0,
    "O2-": 0,
    "F1-": 0,
    "Na1+": 0,
    "Mg2+": 0,
    "Al3+": 0,
    "Siva": 0,
    "Si4+": 0,
    "Cl1-": 0,
    "K1+": 0,
    "Ca2+": 0,
    "Sc3+": 0,
    "Ti2+": 0,
    "Ti3+": 0,
    "Ti4+": 0,
    "V2+": 0,
    "V3+": 0,
    "V5+": 0,
    "Cr2+": 0,
    "Cr3+": 0,
    "Mn2+": 0,
    "Mn3+": 0,
    "Mn4+": 0,
    "Fe2+": 0,
    "Fe3+": 0,
    "Co2+": 0,
    "Co3+": 0,
    "Ni2+": 0,
    "Ni3+": 0,
    "Cu1+": 0,
    "Cu2+": 0,
    "Zn2+": 0,
    "Ga3+": 0,
    "Ge4+": 0,
    "Br1-": 0,
    "Rb1+": 0,
    "Sr2+": 0,
    "Zr4+": 0,
    "Nb3+": 0,
    "Nb5+": 0,
    "Mo3+": 0,
    "Mo5+": 0,
    "Mo6+": 0,
    "Ru3+": 0,
    "Ru4+": 0,
    "Rh3+": 0,
    "Rh4+": 0,
    "Pd2+": 0,
    "Pd4+": 0,
    "Ag1+": 0,
    "Ag2+": 0,
    "Cd2+": 0,
    "In3+": 0,
    "Sn2+": 0,
    "Sn4+": 0,
    "Sb3+": 0,
    "Sb5+": 0,
    "I1-": 0,
    "Cs1+": 0,
    "Ba2+": 0,
    "La3+": 0,
    "Ce3+": 0,
    "Ce4+": 0,
    "Pr3+": 0,
    "Pr4+": 0,
    "Nd3+": 0,
    "Pm3+": 0,
    "Sm3+": 0,
    "Eu2+": 0,
    "Eu3+": 0,
    "Gd3+": 0,
    "Tb3+": 0,
    "Dy3+": 0,
    "Ho3+": 0,
    "Er3+": 0,
    "Tm3+": 0,
    "Yb2+": 0,
    "Yb3+": 0,
    "Lu3+": 0,
    "Hf4+": 0,
    "Ta5+": 0,
    "W6+": 0,
    "Os4+": 0,
    "Ir3+": 0,
    "Ir4+": 0,
    "Pt2+": 0,
    "Pt4+": 0,
    "Au1+": 0,
    "Au3+": 0,
    "Hg1+": 0,
    "Hg2+": 0,
    "Tl1+": 0,
    "Tl3+": 0,
    "Pb2+": 0,
    "Pb4+": 0,
    "Bi3+": 0,
    "Bi5+": 0,
    "Ra2+": 0,
    "Ac3+": 0,
    "Th4+": 0,
    "U3+": 0,
    "U4+": 0,
    "U6+": 0,
    "Np3+": 0,
    "Np4+": 0,
    "Np6+": 0,
    "Pu3+": 0,
    "Pu4+": 0,
    "Pu6+": 0,
}

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
