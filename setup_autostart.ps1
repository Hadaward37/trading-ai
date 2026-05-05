# setup_autostart.ps1 -- Execute como Administrador (botao direito -> "Run as administrator")
# Registra Trading AI para iniciar automaticamente no login do Windows

$projectDir = "C:\Users\dudut\trading-ai"
$batFile    = Join-Path $projectDir "start_trading_ai.bat"
$taskName   = "TradingAI Startup"

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host " Trading AI -- Configurando Autostart" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Verifica se o bat existe
if (-not (Test-Path $batFile)) {
    Write-Host "ERRO: Arquivo nao encontrado: $batFile" -ForegroundColor Red
    exit 1
}

# Remove tarefa existente se houver
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

# Cria a tarefa
$action   = New-ScheduledTaskAction -Execute $batFile -WorkingDirectory $projectDir
$trigger  = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0) `
    -MultipleInstances  IgnoreNew `
    -StartWhenAvailable $true

Register-ScheduledTask `
    -TaskName   $taskName `
    -Action     $action `
    -Trigger    $trigger `
    -Settings   $settings `
    -Description "Inicia Trading AI (scheduler Telegram + dashboard Streamlit) ao fazer login" `
    -RunLevel   Highest `
    -Force

if ($?) {
    Write-Host " Tarefa '$taskName' registrada com sucesso!" -ForegroundColor Green
    Write-Host " O Trading AI iniciara automaticamente no proximo login." -ForegroundColor Green
    Write-Host ""
    Write-Host " Para verificar:" -ForegroundColor Yellow
    Write-Host "   Get-ScheduledTask -TaskName '$taskName'" -ForegroundColor Yellow
    Write-Host ""
    Write-Host " Para remover o autostart:" -ForegroundColor Yellow
    Write-Host "   Unregister-ScheduledTask -TaskName '$taskName' -Confirm:`$false" -ForegroundColor Yellow
} else {
    Write-Host " ERRO ao registrar a tarefa. Execute como Administrador." -ForegroundColor Red
}

Write-Host ""
