from PyInstaller.utils.hooks import collect_data_files, collect_submodules


hiddenimports = collect_submodules("gridstate")
datas = collect_data_files("gridstate")
