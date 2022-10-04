from typing import NoReturn

def requires_package(package: str,
                     name: str) -> NoReturn:
    from importlib.util import find_spec
    if find_spec(package) is None:
        print(f'Error: {name} not installed. If installing with pip,',
              'use "pip install pynuml[torch] to install PyTorch dependencies.')
        import sys
        sys.exit(1)

def requires_torch() -> NoReturn:
    requires_package('torch', 'PyTorch')

def requires_pyg() -> NoReturn:
    requires_package('torch_geometric', 'PyG')

