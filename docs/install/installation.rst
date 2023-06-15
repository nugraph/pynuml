Installation
============

In order to best make use of the **pynuml** package, it is strongly encouraged to install dependencies with Anaconda before installing **pynuml**. Parallel processing functionality requires an MPI installation, which will be automatically configured when you install the `numl` conda environment.

Installing dependencies
-----------------------

 An environment file for conda is available `here`_., and can be installed using::
    conda env create -f numl.yml

Once installed, this environment will need to be activated at the start of each terminal session::
    conda activate numl

.. _here: https://raw.githubusercontent.com/vhewes/numl-docker/main/numl.yml

Installing pynuml
-----------------

**pynuml** is available on PyPi, and can be installed easily using::
    pip install pynuml

If you're setting up the repository for development, you can also clone it and install it in editable mode::
    git clone https://github.com/vhewes/pynuml
    pip install -e ./pynuml

If installed in editable mode, any changes made to the package will instantaneously be reflected when the module is imported in Python.