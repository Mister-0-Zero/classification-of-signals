# Third party components

This distribution uses Python, Qt for Python (PySide6 Essentials), Shiboken6,
Qt 6 and the PyInstaller bootloader.

Qt for Python and Shiboken are distributed under their upstream LGPL/GPL
or commercial license terms. See the bundled upstream license texts under
LICENSES/ and the Qt for Python project:
https://code.qt.io/cgit/pyside/pyside-setup.git/

The portable package keeps Qt libraries as separate dynamic libraries in
_internal/. No application-specific changes were made to those libraries.

PyInstaller's bootloader has an exception permitting distribution of the
generated application under the application's own license. See upstream:
https://pyinstaller.org/en/stable/license.html

Python is distributed under the PSF license:
https://docs.python.org/3/license.html

The application source is located in desktop/ in the classification-of-signals
repository. This notice does not grant licenses beyond the upstream terms.
