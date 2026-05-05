@echo off
echo.
echo  ================================================
echo   Trading AI ^| Encerrando processos...
echo  ================================================
echo.

:: Para pelo titulo da janela (mais cirurgico)
echo  Encerrando Scheduler...
taskkill /FI "WINDOWTITLE eq TradingAI-Scheduler" /T /F >nul 2>&1

echo  Encerrando Dashboard...
taskkill /FI "WINDOWTITLE eq TradingAI-Dashboard" /T /F >nul 2>&1

:: Fallback: encerra qualquer python restante do projeto
echo  Verificando processos Python restantes...
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /NH 2^>nul ^| findstr python') do (
    taskkill /PID %%i /T /F >nul 2>&1
)

:: Encerra qualquer processo streamlit orphan
taskkill /FI "IMAGENAME eq streamlit.exe" /T /F >nul 2>&1

echo.
echo  ================================================
echo   Trading AI parado!
echo  ================================================
echo.
