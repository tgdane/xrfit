from distutils.core import setup

packages=[
	'',
	'xrfit',
]
package_dir = {'':'lib'}

setup(
	name          =   "xrfit",
	version       =   "0.1.0",
	description   =   "Fitting routines for X-ray diffraction peaks",
	author        =   "Thomas Dane",
	author_email  =   "dane@esrf.fr",
	packages      =   packages,
	package_dir   =   package_dir,
)
