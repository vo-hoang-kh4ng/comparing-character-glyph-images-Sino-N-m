def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "data: needs final_characteristics-v2.xlsx and openpyxl (run: pip install openpyxl)",
    )
