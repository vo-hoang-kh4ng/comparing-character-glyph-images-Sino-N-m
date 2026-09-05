# ``submission/`` là bản sao mã nguồn do ``package_submission.py`` sinh ra, và nó chứa cả các file
# test. Không loại trừ thì pytest gom cả hai bản, rồi dừng với "import file mismatch" vì hai file
# khác đường dẫn lại cùng tên module. Lỗi này tái diễn sau mỗi lần đóng gói nên phải chặn ở đây,
# chứ không phải đi xoá __pycache__ mỗi lần.
collect_ignore_glob = ["submission/*", "build/*", "dist/*"]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "data: needs final_characteristics-v2.xlsx and openpyxl (run: pip install openpyxl)",
    )
