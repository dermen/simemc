
def pytest_addoption(parser):
    parser.addoption("--ndev", action="store", default="1")


def pytest_generate_tests(metafunc):
    ndev = int(metafunc.config.option.ndev)
    if 'ndev' in metafunc.fixturenames and ndev is not None:
        metafunc.parametrize("ndev", [ndev])
