import sys; sys.path.insert(0, ".")
from core.pythex_bridge import explain_signal
r = explain_signal("BUY", 1.16858, 1.16579, 1.17304, 46.5)
print(f"Chars: {len(r)}")
print(f"Cortado: {r.endswith('...')}")
print(f"Frase completa: {r[-1] in '.!?'}")
print(f"Texto: {r}")
