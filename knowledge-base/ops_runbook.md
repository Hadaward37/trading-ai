# Ops Runbook — Sistema 1 (trading-ai)

**Criado:** 22/05/2026  
**Contexto:** Oracle VM.Standard.E2.1.Micro (1 OCPU / 1 GB RAM + 1 GB swap)

---

## Acesso SSH

```bash
ssh -i ~/.ssh/ssh-key-2026-05-05.key ubuntu@137.131.228.166
# Windows PowerShell:
ssh -i "$env:USERPROFILE\.ssh\ssh-key-2026-05-05.key" ubuntu@137.131.228.166
```

---

## Healthcheck rápido (uso diário durante incubação)

```bash
ssh oracle '
  echo "=== HEARTBEAT ===" && tail -1 ~/trading-ai/logs/heartbeat.txt
  echo "=== SINAIS ===" && wc -l ~/trading-ai/logs/signals.jsonl 2>/dev/null || echo "0"
  echo "=== RAM ===" && free -h | grep Mem
  echo "=== SWAP ===" && free -h | grep Swap
  echo "=== SERVICE ===" && systemctl is-active trading-ai.service
'
```

---

## Comandos de diagnóstico

```bash
# Ver últimas linhas do log do scheduler
tail -50 ~/trading-ai/scheduler.log

# Ver últimos 5 sinais gerados
tail -5 ~/trading-ai/logs/signals.jsonl | python3 -m json.tool

# Verificar outcomes preenchidos
grep -c '"outcome_1h": null' ~/trading-ai/logs/signals.jsonl  # pendentes
grep -cE '"outcome_1h": -?[0-9]' ~/trading-ai/logs/signals.jsonl  # preenchidos

# Status de todos os services relevantes
systemctl list-units --type=service --all | grep -iE "trading|polymarket"

# Uso de memória por processo
ps aux --sort=-%mem | head -8

# Ver log do cron do outcome_filler
tail -20 ~/trading-ai/logs/outcome_filler.log

# Ver crontab atual
crontab -l
```

---

## Reativação pós-06/07 (executar NA ORDEM)

### Passo 1 — Verificar saúde antes de qualquer mudança

```bash
cd ~/trading-ai
git log --oneline -3
free -h
systemctl status trading-ai.service --no-pager | head -10
wc -l logs/signals.jsonl
```

### Passo 2 — Backup do dataset de incubação

```bash
cp ~/trading-ai/logs/signals.jsonl ~/backups/signals_incubation_final_$(date +%Y%m%d).jsonl
tar -czf ~/backups/incubation-v1.0-$(date +%Y%m%d).tar.gz \
    ~/trading-ai/logs/ \
    ~/trading-ai/data/trading.db \
    ~/trading-ai/scheduler.log
ls -lah ~/backups/
```

### Passo 3 — Git pull do código novo (somente após decisão)

```bash
cd ~/trading-ai
git fetch --all --tags
git pull origin master
git log --oneline -3
```

### Passo 4 — Validar config nova antes de restart

```bash
source ~/trading-ai/venv/bin/activate
python3 -c "import config; print('ASSETS:', list(config.ASSETS.keys())); print('VERSION:', config.STRATEGY_VERSION)"
python3 -c "from core import scheduler, paper_trading, signal_logger; print('IMPORTS OK')"
```

### Passo 5 — Restart do trading-ai com nova config

```bash
sudo systemctl stop trading-ai.service
sleep 3
sudo systemctl start trading-ai.service
sleep 8
sudo systemctl status trading-ai.service --no-pager
tail -20 ~/trading-ai/scheduler.log
```

### Passo 6 — Reativar observator (somente se VM for A1.Flex)

```bash
# PRÉ-REQUISITO: upgrade da VM para A1.Flex ARM (24 GB RAM)
# Débito técnico: criar trading-ai-observator.service antes deste passo
# Ver: knowledge-base/edge_thesis.md seção "Débito técnico"

# Por enquanto (VM micro): NÃO reativar o observator
# sudo systemctl enable trading-ai-observator.service
# sudo systemctl start trading-ai-observator.service
```

### Passo 7 — Reativar polymarket (somente se VM for A1.Flex)

```bash
# PRÉ-REQUISITO: upgrade da VM para A1.Flex ARM
sudo systemctl enable polymarket-bot.service
sudo systemctl start polymarket-bot.service
sudo systemctl enable polymarket-dashboard.service
sudo systemctl start polymarket-dashboard.service
sleep 5
free -h  # confirmar RAM ainda OK
```

---

## Troubleshooting

### SSH não conecta / timeout

```bash
# Tentar via Oracle Cloud Console (fallback sem SSH):
# https://cloud.oracle.com → Compute → Instances → trading-ai → Console Connection

# Se VM travada (OOM): reboot via console
# Após reboot: verificar que trading-ai.service subiu automaticamente
systemctl status trading-ai.service
```

### Trading-ai crashou / não está rodando

```bash
# Verificar causa no journal
journalctl -u trading-ai.service --no-pager -n 50

# Ver scheduler.log (onde a maioria dos erros aparecem)
tail -100 ~/trading-ai/scheduler.log | grep -iE "error|exception|traceback"

# Restart manual
sudo systemctl restart trading-ai.service
sleep 10
tail -20 ~/trading-ai/scheduler.log
```

### Swap sendo consumido (sinal de pressão de memória)

```bash
free -h
# Se Swap Used > 0: investigar antes de reiniciar serviços
ps aux --sort=-%mem | head -5
# Candidato óbvio: polykmarket ou observator órfão
ps aux | grep -iE "polymarket|observator" | grep -v grep
# Matar se necessário:
# sudo pkill -f polymarket
```

### yfinance retornando erro consistente

```bash
# Testar manualmente:
cd ~/trading-ai && source venv/bin/activate
python3 -c "
import yfinance as yf
t = yf.Ticker('VALE3.SA')
print(t.fast_info.get('last_price'))
"
# Se falhar: verificar rate limiting ou mudança de API do yfinance
# pip show yfinance  # ver versão atual
# pip install --upgrade yfinance  # atualizar (somente se necessário)
```

### signals.jsonl vazio ou sem crescimento

```bash
# 1. Verificar se há sinais sendo gerados no log
grep "Signal=" ~/trading-ai/scheduler.log | tail -20

# 2. Verificar se há BUY/SELL (não apenas HOLD)
grep -iE "BUY|SELL" ~/trading-ai/scheduler.log | tail -20

# 3. Verificar threshold (sinais rejeitados são logados)
grep "NOT executed" ~/trading-ai/scheduler.log | tail -20
```

---

## Upgrade da VM (roadmap pós-06/07)

### De VM.Standard.E2.1.Micro para A1.Flex ARM

1. Oracle Cloud Console → Compute → Instances → Stop instance
2. Change shape → VM.Standard.A1.Flex → 2 OCPU / 12 GB RAM (grátis no Always Free)
3. Start instance
4. Verificar que IP público não mudou (ou atualizar DNS)
5. SSH e confirmar `free -h` mostra ~12 GB
6. `sudo swapon --show` — swap ainda ativo (manter, não prejudica)
7. Reativar polymarket e observator com MemoryLimit no unit file

---

## Referências rápidas

| Item | Valor |
|---|---|
| IP Oracle | 137.131.228.166 |
| Chave SSH | `~/.ssh/ssh-key-2026-05-05.key` |
| Logs do scheduler | `~/trading-ai/scheduler.log` |
| Heartbeat | `~/trading-ai/logs/heartbeat.txt` |
| Sinais | `~/trading-ai/logs/signals.jsonl` |
| Outcome filler log | `~/trading-ai/logs/outcome_filler.log` |
| Backups | `~/backups/` |
| Swap | `/swapfile` (1 GB, persistido em /etc/fstab) |
