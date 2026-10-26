"""Exibição amigável das respostas nos notebooks."""
import pandas as pd
from IPython.display import Markdown, display

from .rag import RAGResponse

STATUS_LABEL = {
    "respondida": "✅ Respondida com evidências",
    "fora_do_escopo": "🚫 Fora do escopo da base",
    "sem_evidencias": "⚠️ Sem evidências suficientes na recuperação",
    "evidencias_insuficientes": "⚠️ Evidências insuficientes (avaliação do modelo)",
}


def show(resp: RAGResponse, show_queries: bool = True, max_evidences: int = 15):
    md = [f"### ❓ {resp.question}", f"**Status:** {STATUS_LABEL.get(resp.status, resp.status)}"]
    if resp.filters_applied:
        md.append(f"**Filtro de metadados aplicado:** `{resp.filters_applied}`")
    if show_queries and resp.retrieval is not None:
        md.append("**Consultas usadas na busca:** " + " · ".join(f"`{q}`" for q in resp.retrieval.queries))
        s = resp.retrieval.support
        if s.get("avaliacoes_relacionadas"):
            md.append(
                f"**Perfil do tema:** {'≥' if s.get('limite_da_amostra') else ''}{s['avaliacoes_relacionadas']} avaliações "
                f"relacionadas · nota média {s['nota_media']} · {s['pct_notas_1_2']}% com nota 1-2"
            )
    if resp.model_used:
        md.append(f"**Modelo de geração:** `{resp.model_used}`")
    md += ["---", resp.answer]
    if resp.limitations:
        md.append(f"\n> **Limitações:** {resp.limitations}")
    if resp.invalid_refs:
        md.append(f"\n> ⚠️ Citações inválidas descartadas: {resp.invalid_refs}")
    display(Markdown("\n\n".join(md)))

    table = resp.evidence_table(only_cited=bool(resp.cited_refs))
    if not table.empty:
        display(Markdown(f"**Evidências {'citadas' if resp.cited_refs else 'recuperadas'} ({len(table)}):**"))
        with pd.option_context("display.max_colwidth", 200):
            display(table.drop(columns=["citada", "review_id"]).head(max_evidences).style.hide(axis="index"))
