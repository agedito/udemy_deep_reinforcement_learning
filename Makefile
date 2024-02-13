install: install-packages install-roms

install-packages:
	pip install -r requirements.txt
	pip install swig
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	pip install swig Box2d gym[all]


install-roms:
	python .\.venv\Lib\site-packages\retro\scripts\import_path.py '.\course_resources\Roms\HC ROMS\BY ALPHABET\'
	python .\.venv\Lib\site-packages\retro\scripts\import_path.py '.\course_resources\Roms\HC ROMS\BY ALPHABET (PAL)\'
	python .\.venv\Lib\site-packages\retro\scripts\import_path.py '.\course_resources\Roms\HC ROMS\BY COMPANY\'
	python .\.venv\Lib\site-packages\retro\scripts\import_path.py '.\course_resources\Roms\HC ROMS\BY COMPANY (PAL)\'
	python .\.venv\Lib\site-packages\retro\scripts\import_path.py '.\course_resources\Roms\HC ROMS\NTSC VERSIONS OF PAL ORIGINALS\'
	python .\.venv\Lib\site-packages\retro\scripts\import_path.py '.\course_resources\Roms\HC ROMS\PAL VERSIONS OF NTSC ORIGINALS\'
	python .\.venv\Lib\site-packages\retro\scripts\import_path.py '.\course_resources\Roms\ROMS\'