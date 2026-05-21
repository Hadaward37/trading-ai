#!/bin/bash
# Validação rápida do estado do Sistema 1 na Oracle
# Uso: bash scripts/oracle_healthcheck.sh

cd /home/ubuntu/trading-ai

echo "=== HEARTBEAT ==="
if [ -f logs/heartbeat.txt ]; then
    cat logs/heartbeat.txt
    LAST=$(cat logs/heartbeat.txt)
    NOW=$(date -u +%s)
    LAST_TS=$(date -d "$LAST" +%s 2>/dev/null)
    if [ -n "$LAST_TS" ]; then
        DIFF=$(( (NOW - LAST_TS) / 60 ))
        echo "Última atualização: $DIFF minutos atrás"
    fi
else
    echo "[ERRO] heartbeat.txt não existe"
fi

echo ""
echo "=== SINAIS LOGADOS ==="
if [ -f logs/signals.jsonl ]; then
    TOTAL=$(wc -l < logs/signals.jsonl)
    echo "Total: $TOTAL sinais"
    NULL_5MIN=$(grep -c '"outcome_5min": null' logs/signals.jsonl)
    FILLED_5MIN=$(grep -cE '"outcome_5min": -?[0-9]' logs/signals.jsonl)
    echo "outcome_5min: $FILLED_5MIN preenchidos / $NULL_5MIN pendentes"
else
    echo "Nenhum sinal logado ainda"
fi

echo ""
echo "=== CRON OUTCOME FILLER ==="
if [ -f logs/outcome_filler.log ]; then
    tail -5 logs/outcome_filler.log
else
    echo "Cron ainda não rodou (esperar 5min após deploy)"
fi

echo ""
echo "=== SCHEDULER PROCESS ==="
ps aux | grep -E 'core.scheduler' | grep -v grep
