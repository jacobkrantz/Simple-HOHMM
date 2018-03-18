Getting Started
===============

Installation for Python 2 or 3
------------------------------

``Simple-HOHMM`` can be installed directly from Github using ``pip``. You must have ``git`` installed for this process to work.
::

	>>> pip install git+https://github.com/jacobkrantz/Simple-HOHMM.git

If you want the most recent staging build:
::

	>>> pip install git+https://github.com/jacobkrantz/Simple-HOHMM.git@staging

Alternative: to view the source code and run the tests before installation:
::

	>>> git clone https://github.com/jacobkrantz/Simple-HOHMM.git
	>>> cd Simple-HOHMM
	>>> python setup.py test
	>>> python setup.py install

Installation for Pypy
---------------------

For usage with ``pypy``, you must install with ``pip`` inside ``pypy``:
::

	>>> pypy -m pip install git+https://github.com/jacobkrantz/Simple-HOHMM.git

If this fails, try installing ``pip`` for ``pypy`` first:
::

	>>> curl -O https://bootstrap.pypa.io/get-pip.py
	>>> pypy get-pip.py

If you want the most recent staging build still with ``pypy``:
::

 	>>> pypy -m pip install git+https://github.com/jacobkrantz/Simple-HOHMM.git@staging

Alternative staging branch with ``pypy``:
::

	>>> sudo pypy -m pip install --upgrade https://github.com/jacobkrantz/Simple-HOHMM/archive/staging.zip
