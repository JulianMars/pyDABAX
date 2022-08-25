pyDABAX
=======


pyDABAX aims to make the dabax database fast and easy accessible in python. Besides the access to the original
database, pyDABAX also provides high level functionality for important quantities like anomalous x-ray and neutron
form-factors, absorption edges, and compton scattering.


Installation
============
Package
-------------------
Install with pip into your current environment.

```bash
pip install pyDABAX
```

The following dependencies will be installed by pip:

- `numpy <https://www.numpy.org/>`
- `TinyDB <https://github.com/msiemens/tinydb>`
- `astropy <https://github.com/astropy/astropy>`
- `pandas`
- `regex`

Manual installation
----------------------
Clone the current git repository:

```bash
# Run in your terminal or conda terminal
git clone https://github.com/JulianMars/pyDABAX.git
```

You can install pyDABAX from inside the git folder to your current environment using:

```bash
# Install package using pip
cd ./pyDABAX.git           # Change into the pyDABAX.git folder
pip install .              # Use the pip package manager to install pyDABAX in your current python environment
```


High-level interface
====================

Getting Started 
---------------
Create compound from string with fixed energy.
```python
from pydabax import *
Gold = Compound('Au', energy='10 keV', density='element')
```

Obtain refractive index, x-ray form factor, and attenuation coefficient.
```python
print('Refractive index: δ + βj = {:.2e}'.format(Gold.deltabeta))
print('Formfactor: f = {:.1f}'.format(Gold.f))
print('Attenuation coefficient: mu = {:.3f}'.format(Gold.mu))
```
> Refractive index: δ + βj = 2.99e-05+2.21e-06j
> Formfactor: f = 73.4+5.4j
> Attenuation coefficient: mu = 2218.580 1 / cm

In jupyter notebooks Compounds and Elements have a html representation with useful parameters:
```python
from pydabax import *
Elements['O']
```
<h1>Oxygen</h1><table> <tr> <th>Symbol</th> <td>O</td> </tr><tr> <th>Atomic number</th> <td>8</td> </tr><tr> <th>Atomic mass</th> <td>15.9994 u</td> </tr><tr> <th>Charge</th> <td>0</td> </tr><tr> <th>Atomic radius</th> <td>0.65 Angstrom</td> </tr><tr> <th>Covalent radius</th> <td>0.73 Angstrom</td> </tr><tr> <th>Melting point</th> <td>50.35 K</td> </tr><tr> <th>Boiling point</th> <td>90.18 K</td> </tr><tr> <th>Energy</th> <td>8.047 keV</td> </tr><tr> <th>q</th> <td>0.0 1 / Angstrom</td> </tr><tr> <th>X-ray formfactor</th> <td>8.052 electron</td> </tr><tr> <th>K<sub>α1</sub></th> <td>0.5249 keV</td> </tr><tr> <th>K<sub>α2</sub></th> <td>0.5249 keV</td> </tr><tr> <th>K<sub>β</sub></th> <td>-</td> </tr><tr> <th>b<sub>coh</sub></th> <td>(5.803+0j) fm</td> </tr><tr> <th>b<sub>inc</sub></th> <td>-</td> </tr><tr> <th>σ<sub>coh</sub></th> <td>4.232 barn</td> </tr><tr> <th>σ<sub>inc</sub></th> <td>0.0008 barn</td> </tr><tr> <th>absorption (2200m/s)</th> <td>0.0002 barn</td> </tr></table>


Plot the q-dependent Form factor density
----------------------------------------
```python
import matplotlib.pyplot as plt
import numpy as np
from pydabax import Compound
#q-space
q = np.linspace(0, 35, 101)
#Create Compounds
Gold = Compound("Au", energy="8.047 keV", density="element")
Water = Compound("H2O", energy="8047 eV", density="997 kg/m^3")
Il = Compound('(CH6N)0.4(C8H15N2)0.6(CF3SO2)2N', density="mcgowan") 
#Set q of compounds
Water.q = q
Gold.q = q
Il.q = q
#Prepare plot
fig, ax = plt.subplots(figsize=[3.375, 3])
ax.set_xlabel("q (1/Å)")
ax.set_ylabel("f1 / V (e/Å)")
#Obtain f from compounds and plot
ax.plot(Water.q, Water.f.real/Water.molecular_volume, label="H2O at 8.047 keV")
ax.plot(Gold.q, Gold.f.real/Gold.molecular_volume, label="Gold at 8.047 keV")
ax.plot(Il.q, Il.f.real/Il.molecular_volume, label="Ionic Liquid at 8.047 keV")
_ = ax.legend(prop={"size": 8})
```

<img src="./blob/formfactor.jpg" alt="formfactor" width="450"/>

Ions and Isotopes
------------------
pydabax supports all common isotopes and ions and fractional formulas. Compounds can be multiplied and added.

```python
Compound('2H2O', density="mcgowan") 
Compound('OH-', density="mcgowan") 
Compound('YB2Cu3O6.93', density="element") 
#create 0.8 mol/kg aqueous CsCl solution
cp = 0.8 * Compound('CsCl') + 55.555 * Compound(H2O)
```

Units
-----
As the different flavors of x-ray analysis prefers different units, pyDABAX uses astropy to handle physical quantities
consisting of a value and a unit. Hence, unit handling should be flexible and coherent within the package.
First, set the preferred global units. Standard units are keV, Å, 1/Å, and °.
All inputs without explicitly specified unit and all outputs will have this unit.

```python
#Photon energy
UnitSettings.UNIT_E = 'eV'
#Momentum transfer
UnitSettings.UNIT_Q = '1/nm'
#Wavelength
UnitSettings.UNIT_R = 'nm'
#Total scattering angles
UnitSettings.UNIT_TTH = 'rad'
```

Accessing the X-ray database dabax
==================================

Return a list of all available symbols:
```python
import pydabax as dbx
dbx.get_symbols()
```

Show all available entries for carbon.
```python
import pydabax as dbx
dbx.get_keys("C")
```
>['atomic_number',
'symbol',
'element_symbol',
'name',
'charge',
'mass_number',
'mcgowan_volume',
'atomic_weight',
'nist_f1f2_chantler',
'nist_edges_chantler',
'cxro_f1f2_henke',
'nist_b_sears',
'dabax_AtomicConstants',
'dabax_ComptonProfiles',
'dabax_CrossSec_BrennanCowan',
'dabax_CrossSec_Compton_IntegrHubbell',
... ... ...,
'dabax_isf_xop_biggs_linap',
'dabax_JumpRatio_Elam',
'dabax_Neutron_SLCS_DataBooklet',
'dabax_Neutron_SLCS_NeutronNews',
'dabax_RadiativeRates_KrauseScofield',
'dabax_RadiativeRates_L_Scofield',
'dabax_XAFS_McKale_K-edge_R=2.5_A',
'dabax_XAFS_McKale_K-edge_R=4.0_A',
'dabax_XAFS_McKale_L-edge_R=2.5_A',
'dabax_XAFS_McKale_L-edge_R=4.0_A',
'dabax_XREmission_NIST',
'dabax_XREmission',
'dabax_XREmissionWeights',
'mcgowan_vol']

Get the CXRO Henke table for f1 and f2.
```python
dbx.get_dabax("C", "cxro_f1f2_henke")
```

>E (eV)	f1	f2
0	10.0000	-9999.00000	0.806885
1	10.1617	-9999.00000	0.851522
2	10.3261	-9999.00000	0.898628
3	10.4931	-9999.00000	0.948341
4	10.6628	-9999.00000	1.000800
...	...	...	...
497	28135.1000	6.00026	0.000515
498	28590.2000	6.00020	0.000496
499	29052.6000	6.00013	0.000478
500	29522.5000	6.00007	0.000460
501	30000.0000	6.00000	0.000443
502 rows × 3 columns


The database file is in json format and can be thus viewed with all common json viewers.
Jupyter lab comes with an integrated json viewer.  