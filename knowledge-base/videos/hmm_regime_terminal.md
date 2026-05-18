# AI Pathways — HMM Regime Terminal com Claude Code

## Conceito Principal
Hidden Markov Models (HMM) para detecção de regimes de mercado.
Usado por fundos institucionais como Renaissance Technologies (Jim Simons).

## O que o HMM faz
- Classifica mercado em regimes (ex: Bull Run, Crash, Lateral, Chop)
- Baseado em distribuição Gaussiana e probabilidade pura
- NÃO tenta prever preço — mapeia ESTADO do mercado
- Biblioteca Python: hmmlearn

## Arquitetura "Autenticação em 2 Etapas"
Etapa 1: HMM detecta regime (Bull, Bear, Lateral)
Etapa 2: Só entra se regime correto + 7/8 confirmações técnicas

## Técnicas Importantes

### Signal Hysteresis
- Tempo mínimo de espera antes de aceitar mudança de regime
- Evita troca de regime a cada candle de ruído
- Reduz stops seguidos por falsos sinais

### Confidence Score
- Probabilidade estatística do regime (ex: 92%)
- Se confiança baixa → sistema trava
- JÁ IMPLEMENTADO no Event Observer (conf=0.662)

### Hard Cooldown
- Proíbe reentrada nas próximas 48h após saída de trade
- Protege capital contra volatilidade pós-movimento

## Como Aplicar no Projeto
- Localização: research/fractal_lab/regimes/ ou app/regimes/detector.py
- Cruzar HMM com Coerência Fractal (15s e 1m)
- Implementar Signal Hysteresis no Observer
- 7 estados de regime (igual ao vídeo)

## Relevância pós 06/06
- Prioridade 3 no roadmap
- Após: validação Sistema 1 + Adaptive Sizing
- Substituir regime classifier v3 atual por HMM
- Biblioteca: pip install hmmlearn

## Status de Implementação
Implementado em: research/fractal_lab/regimes/hmm_detector.py
Data: 17/05/2026

### Resultados Holdout BBDC4 1H
- Confiança média: 95.8%
- Barras travadas por baixa confiança: 1.2%
- Estabilidade de regime: 87.7%
- Rule-based: 64.8% TRENDING (thresholds fixos)
- HMM: redistribuição mais equilibrada (aprendida dos dados)

### Como rodar
```powershell
.\venv\Scripts\python -X utf8 -m research.fractal_lab.regimes.hmm_runner
.\venv\Scripts\python -X utf8 -m research.fractal_lab.regimes.hmm_runner --states 7
```

### Vantagem sobre rule-based atual
Rule-based = thresholds fixos manuais
HMM = aprende distribuição Gaussiana dos dados automaticamente
