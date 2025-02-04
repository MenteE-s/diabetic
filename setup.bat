@echo off
echo Creating Python virtual environment...
python -m venv env

echo Activating virtual environment...
call env\Scripts\activate.bat

echo Creating Logs folder...
mkdir logs 2>nul

echo Installing requirements...
pip install -r requirements.txt

echo Setup completed!