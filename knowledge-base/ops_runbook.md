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

## Migração para A1.Flex ARM (13/06/2026 — CORRIGIDO)

> ⚠️ **NÃO é "Change Shape".** A VM atual é E2.1.Micro **x86_64**; A1.Flex é
> **ARM (aarch64)**. Oracle não troca arquitetura in-place (o boot volume x86
> não dá boot em ARM, e o A1.Flex nem aparece no "Edit shape" de uma instância
> x86). **É criar uma instância nova ARM e migrar.**
>
> Motivo da migração: rodar o ensemble ML (tensorflow/LSTM) que não cabe em 1GB.
> Ver `knowledge-base/incident_20260613.md`.

### Parte A — no Console OCI (só o usuário faz)
1. Compute → Instances → Create instance
2. Shape: **VM.Standard.A1.Flex** (Ampere). Always Free: até 4 OCPU / 24 GB.
   - ⚠️ A1 free vive dando "Out of capacity" — pode precisar tentar outro AD/horário.
3. Image: **Ubuntu 22.04 (aarch64/ARM)**
4. Mesma VCN/subnet; **mesma chave SSH pública** (`ssh-key-2026-05-05.key`)
5. IP: a nova instância vem com IP novo. Opção simples: usar o IP novo e
   atualizar este runbook. (O 137.131.228.166 é ephemeral da instância antiga.)
6. Anotar o IP público novo e me passar.

### Parte B — setup do servidor (eu faço via SSH, com o IP novo)
```bash
sudo apt update && sudo apt install -y python3-venv python3-dev build-essential git
cd ~ && git clone https://github.com/Hadaward37/trading-ai.git
cd trading-ai && python3 -m venv venv
./venv/bin/pip install -U pip wheel
./venv/bin/pip install -r requirements.txt      # agora inclui tensorflow/joblib/sklearn
mkdir -p logs backups
```
Depois, do local, enviar segredos + modelos + histórico:
```powershell
$ip = "<IP_NOVO>"
scp -i $key .env ubuntu@${ip}:~/trading-ai/.env
scp -i $key models/lstm_eurusd_1h.h5 models/lstm_scaler.pkl `
            models/xgboost_eurusd.pkl models/xgboost_scaler.pkl ubuntu@${ip}:~/trading-ai/models/
scp -i $key data/trading.db ubuntu@${ip}:~/trading-ai/data/   # opcional: preservar histórico
```
Recriar systemd unit (`/etc/systemd/system/trading-ai.service`, ExecStart=
`venv/bin/python run_scheduler.py`, Restart=always) e crontab:
```cron
*/5 * * * * cd /home/ubuntu/trading-ai && /home/ubuntu/trading-ai/venv/bin/python scripts/fill_outcomes_job.py >> /home/ubuntu/trading-ai/logs/outcome_filler.log 2>&1
# Backup CORRIGIDO (o antigo copiava signals.jsonl que não existe — pipeline usa SQLite):
0 3 * * * cp /home/ubuntu/trading-ai/data/trading.db /home/ubuntu/backups/trading_$(date +\%Y\%m\%d).db 2>>/home/ubuntu/trading-ai/logs/backup.log
```

### Parte C — verificar
```bash
sudo systemctl restart trading-ai.service && sleep 85
grep -iE 'lstm|xgboost' scheduler.log | tail   # NÃO deve dizer "not found"/"No module"
grep -E 'EUR/USD|signals_1h' scheduler.log | tail
```
LSTM e XGBoost devem carregar; com o cérebro vivo o portão de consenso passa a
deixar sinais reais (BUY/SELL) quando as condições baterem. Confirmar
`Saved ... signals_1h` com datetime recente.

### Parte D — desligar a instância antiga
Só **depois** de validar a nova. Stop (não terminate) a E2.1.Micro por alguns
dias como fallback. Atualizar IP neste runbook e no CLAUDE.md.

> **Compat Keras:** o `lstm_eurusd_1h.h5` foi salvo em Keras 2; tensorflow>=2.16
> usa Keras 3 e pode falhar ao carregar `.h5` legado. Se acontecer:
> `./venv/bin/pip install tf-keras` e/ou setar `TF_USE_LEGACY_KERAS=1`, ou
> retreinar o LSTM na VM nova (24GB aguenta). Resolver no passo de verificação.

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
