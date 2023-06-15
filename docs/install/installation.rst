Installation
============

In order to best make use of the **pynuml** package, it is strongly encouraged to install dependencies with Anaconda before installing **pynuml**. Parallel processing functionality requires an MPI installation, which will be automatically configured when you install the `numl` conda environment.

Installing with Anaconda
------------------------

Installing **pynuml** dependencies requires an Anaconda installation that utilises `conda-forge`. If you need to install Anaconda, we recommend using the `Mambaforge`_ variant.

An environment file for conda is available `here`_, and can be installed using::

    conda env create -f numl.yml

Once installed, this environment will need to be activated at the start of each terminal session::

    conda activate numl

.. _Mambaforge: https://github.com/conda-forge/miniforge#mambaforge
.. _here: https://raw.githubusercontent.com/vhewes/numl-docker/main/numl.yml

This environment contains the most recent version of **pynuml** published to PyPi.

Installing for development
--------------------------

If you're installing **pynuml** for development, you can clone the repository directly and install it in editable mode::

    git clone https://github.com/vhewes/pynuml
    pip install -e ./pynuml

If installed in editable mode, any changes made to the package will instantaneously be reflected when the module is imported in Python.