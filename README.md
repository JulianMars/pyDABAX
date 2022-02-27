=======
pyDABAX
=======


pyDABAX aims to make the dabax database fast and easy accessible in python. Besides the access to the original
database, pyDABAX also provides high level functionality for important quantities like anomalous x-ray and neutron
form-factors, absorption edges, and compton scattering.


Installation
===========

Manual installation
----------------------
Clone the current git repository:

```bash
# Run in your terminal or conda terminal
git clone https://github.com/JulianMars/pyDABAX.git
```

We recommend using environments. Create and activate the environment.
You can install pyDABAX from inside the git folder to your current environment using:

```bash
# Install package using pip
cd ./pyDABAX.git           # Change into the pyDABAX.git folder
pip install .              # Use the pip package manager to install pyDABAX in your current python environment
```

The following dependencies will be installed by pip:

-  `python <https://www.python.org/>`_
-  `numpy <https://www.numpy.org/>`_
-  `TinyDB <https://github.com/msiemens/tinydb>`_
-  `astropy <https://github.com/astropy/astropy>`_

Usage
=====

High-level interface
--------------------

Create compound from string with fixed energy.
```python
from pydabax.elements import Compound
Gold = Compound('Au', energy='10 keV', density='19.3 g/cm^3')
```

Obtain refractive index,
```python
print('Refractive index: δ + βi = {:.3e} + {:.3e}i'.format(Gold.deltabeta.real, Gold.deltabeta.imag))
```
>Refractive index: δ + βi = 2.987e-05 + 2.205e-06i

x-ray form factor,
```python
print('Formfactor: f = {:.3f} + {:.3f}i (e/atom)'.format(Gold.f.real, Gold.f.imag))
```
>Formfactor: f = 73.419 + 5.421i (e/atom)

and attenuation coefficient.
```python
print('Attenuation coefficient: mu = {:.3f} (1/cm)'.format(Gold.mu.value))
```
>Attenuation coefficient: mu = 2218.580 (1/cm)

Plot the q-dependent Form factor:
```python
import matplotlib.pyplot as plt
import numpy as np
from pydabax.elements import Compound

Gold = Compound("Au", energy="8.047 keV", density="19.3 g/cm^3")
Water = Compound("H2O", energy="8047 eV", density="997 kg/m^3")
q = np.linspace(0, 35, 101)

Water.q = q
Gold.q = q
fig, ax = plt.subplots(figsize=[3.375, 3])
ax.set_xlabel("q (1/Å)")
ax.set_ylabel("f1 (e/atom)")

ax.plot(Water.q, Water.f.real, label="H2O at 8.047 keV")
ax.plot(Gold.q, Gold.f.real, label="Gold at 8.047 keV")
_ = ax.legend(prop={"size": 8})
```

Accessing the X-ray database dabax
---------------------------------

Show all available entries for oxygen.
```python
from pydabax.dabax import dabax as dbx
dbx.get_keys("O")
```
Get the CXRO Henke table for f1 and f2.
```python
from pydabax.dabax import dabax as dbx
dbx.get_table("O", "cxro_f1f2_henke")
```

The database file is in json format and can be thus viewed with all common json viewers.
Jupyter lab comes with an integrated json viewer.