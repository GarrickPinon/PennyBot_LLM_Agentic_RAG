@echo off
echo Activating virtual environment...
call venv\Scripts\activate

echo Installing requirements...
pip install --upgrade pip
pip install -r requirements.txt

echo Running ingestion and index build...
python ingest_and_filter.py
python build_index.py

echo Launching agent CLI...
python chat_cli.py
echo All tasks completed.