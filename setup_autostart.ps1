# setup_autostart.ps1 -- Execute como Administrador (botao direito -> "Run as administrator")
# Registra Trading AI para iniciar automaticamente no login do Windows
# Compativel com PowerShell 5.1 (Windows 10/11)

$projectDir = "C:\Users\dudut\trading-ai"
$batFile    = Join-Path $projectDir "start_trading_ai.bat"
$taskName   = "TradingAI Startup"

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host " Trading AI -- Configurando Autostart"          -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Verifica se o bat existe
if (-not (Test-Path $batFile)) {
    Write-Host "ERRO: Arquivo nao encontrado: $batFile" -ForegroundColor Red
    exit 1
}

# Remove tarefa existente se houver (silencioso se nao existir)
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

# Componentes da tarefa agendada
$action  = New-ScheduledTaskAction -Execute $batFile -WorkingDirectory $projectDir
$trigger = New-ScheduledTaskTrigger -AtLogOn

# CORRECAO: -StartWhenAvailable e um switch -- nao recebe $true como argumento
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0) `
    -MultipleInstances  IgnoreNew `
    -StartWhenAvailable

# Registra a tarefa com try/catch para mensagem de erro clara
try {
    Register-ScheduledTask `
        -TaskName    $taskName `
        -Action      $action `
        -Trigger     $trigger `
        -Settings    $settings `
        -Description "Inicia Trading AI (scheduler Telegram + dashboard Streamlit) ao fazer login" `
        -RunLevel    Highest `
        -Force | Out-Null

    Write-Host " OK: Tarefa '$taskName' registrada!" -ForegroundColor Green
    Write-Host " O Trading AI vai iniciar automaticamente no proximo login." -ForegroundColor Green
    Write-Host ""
    Write-Host " Comandos uteis:" -ForegroundColor Yellow
    Write-Host "   Verificar : Get-ScheduledTask -TaskName '$taskName'" -ForegroundColor Yellow
    Write-Host "   Remover   : Unregister-ScheduledTask -TaskName '$taskName' -Confirm:`$false" -ForegroundColor Yellow
    Write-Host "   Executar ja: Start-ScheduledTask -TaskName '$taskName'" -ForegroundColor Yellow
}
catch {
    Write-Host " ERRO: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host " Execute este script como Administrador." -ForegroundColor Red
    exit 1
}

Write-Host ""
