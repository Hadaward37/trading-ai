@echo off
cd /d C:\Users\dudut\trading-ai

echo.
echo  ================================================
echo   Trading AI ^| Inicializando...
echo  ================================================
echo.

:: Cria pasta de logs se nao existir
if not exist logs mkdir logs

:: --------------------------------------------------
:: 1. Scheduler (bot Telegram) em janela minimizada
:: --------------------------------------------------
echo  [1/3] Scheduler (bot Telegram)...
start "TradingAI-Scheduler" /MIN venv\Scripts\python.exe run_scheduler.py

:: Aguarda 3s para o scheduler conectar e enviar mensagem de startup
timeout /t 3 /nobreak > nul

:: --------------------------------------------------
:: 2. Dashboard Streamlit em janela minimizada
:: --------------------------------------------------
echo  [2/3] Dashboard Streamlit (porta 8501)...
start "TradingAI-Dashboard" /MIN venv\Scripts\streamlit.exe run dashboard\app.py --server.port 8501 --server.headless true

:: Aguarda Streamlit iniciar antes de abrir o browser
timeout /t 7 /nobreak > nul

:: --------------------------------------------------
:: 3. Abre navegador no dashboard
:: --------------------------------------------------
echo  [3/3] Abrindo dashboard no navegador...
start "" http://localhost:8501

echo.
echo  ================================================
echo   Trading AI iniciado com sucesso!
echo.
echo   Scheduler : janela "TradingAI-Scheduler"
echo   Dashboard : http://localhost:8501
echo   Logs      : logs\
echo  ================================================
echo.
echo  Use stop_trading_ai.bat para encerrar tudo.
echo.
