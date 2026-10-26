"""Orquestração do pipeline RAG: pergunta -> transformação -> recuperação -> contexto -> resposta -> validação."""
import re
from dataclasses import dataclass, field

import pandas as pd

from . import config, llm
from .filters import build_where
from .retrieval import Evidence, RetrievalResult, Retriever

STATUS_ANSWERED = "respondida"
STATUS_OUT_OF_SCOPE = "fora_do_escopo"
STATUS_NO_EVIDENCE = "sem_evidencias"            # recuperação não encontrou avaliações relevantes o bastante
STATUS_INSUFFICIENT = "evidencias_insuficientes"  # LLM avaliou que o conteúdo recuperado não responde à pergunta

_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
_REF_TOKEN_RE = re.compile(r"[A-Za-z]{1,2}\d+")


def extract_citations(text: str) -> list[str]:
    """Extrai marcadores de citação como [R1], [R2, R5] ou [R3-R4]. Marcadores malformados (ex.: [F11]) também são
    extraídos para serem reportados como inválidos."""
    refs = []
    for group in _BRACKET_RE.findall(text):
        tokens = [t.strip() for t in re.split(r"[,;]", group)]
        if tokens and all(re.fullmatch(r"[A-Za-z]{1,2}\d+(\s*-\s*[A-Za-z]{1,2}\d+)?", t) for t in tokens):
            for t in tokens:
                ends = _REF_TOKEN_RE.findall(t)
                if len(ends) == 2 and ends[0][0].upper() == "R" and ends[1][0].upper() == "R":
                    refs += [f"R{n}" for n in range(int(ends[0][1:]), int(ends[1][1:]) + 1)]
                else:
                    refs += [e.upper() for e in ends]
    return list(dict.fromkeys(refs))


@dataclass
class RAGResponse:
    question: str
    status: str
    answer: str
    evidences: list[Evidence] = field(default_factory=list)
    cited_refs: list[str] = field(default_factory=list)
    invalid_refs: list[str] = field(default_factory=list)
    limitations: str = ""
    analysis: llm.QueryAnalysis | None = None
    retrieval: RetrievalResult | None = None
    context: str = ""
    model_used: str | None = None
    filters_applied: dict | None = None

    @property
    def cited_evidences(self) -> list[Evidence]:
        return [e for e in self.evidences if e.ref in self.cited_refs]

    def evidence_table(self, only_cited: bool = True) -> pd.DataFrame:
        evs = self.cited_evidences if only_cited else self.evidences
        return pd.DataFrame(
            [
                {
                    "ref": e.ref, "citada": e.ref in self.cited_refs, "similaridade": e.similarity,
                    "nota": e.metadata.get("review_score"), "categoria": e.metadata.get("product_category"),
                    "data": e.metadata.get("year_month"), "atraso_dias": e.metadata.get("delivery_delay_days"),
                    "texto_identico_x": e.identical_count, "avaliacao": e.text, "review_id": e.review_id,
                }
                for e in evs
            ]
        )


def _describe(e: Evidence) -> str:
    m = e.metadata
    parts = [f"nota {m['review_score']}/5", f"categoria: {m['product_category']}", f"data: {m['year_month']}"]
    if "delivery_delay_days" in m:
        d = m["delivery_delay_days"]
        parts.append(f"entregue {d} dias após o prazo" if d > 0 else "entregue no prazo")
    elif m.get("order_status") not in (None, "delivered"):
        parts.append(f"status do pedido: {m['order_status']}")
    if e.identical_count > 1:
        parts.append(f"texto idêntico em {e.identical_count} avaliações")
    return "; ".join(parts)


def build_context(retrieval: RetrievalResult, where: dict | None = None) -> str:
    """Contexto enviado ao LLM: recorte aplicado + perfil agregado do tema + avaliações numeradas para citação."""
    lines = []
    if where:
        lines += ["# Recorte aplicado (filtro de metadados)",
                  f"Todas as avaliações abaixo pertencem somente a este recorte: {where}", ""]
    lines.append("# Perfil das avaliações relacionadas ao tema (estimativa a partir da busca semântica)")
    s = retrieval.support
    if s.get("avaliacoes_relacionadas"):
        prefix = "pelo menos " if s.get("limite_da_amostra") else ""
        lines += [
            f"- Avaliações semanticamente relacionadas: {prefix}{s['avaliacoes_relacionadas']}",
            f"- Nota média dessas avaliações: {s['nota_media']}",
            f"- % com nota 1 ou 2: {s['pct_notas_1_2']}%",
            f"- % de pedidos entregues após o prazo (entre os entregues): {s['pct_entregues_com_atraso']}%",
            f"- Categorias mais frequentes: {s['principais_categorias']}",
        ]
    lines.append("\n# Avaliações recuperadas (use os IDs para citar)")
    for e in retrieval.evidences:
        lines.append(f"[{e.ref}] ({_describe(e)}) \"{e.text}\"")
    return "\n".join(lines)


class VoCRAG:
    def __init__(self, retriever: Retriever | None = None, use_query_rewriting: bool = True,
                 use_metadata_filters: bool = True):
        self.retriever = retriever or Retriever()
        self.use_query_rewriting = use_query_rewriting
        self.use_metadata_filters = use_metadata_filters

    def transform_question(self, question: str) -> tuple[llm.QueryAnalysis | None, list[str]]:
        """A pergunta original é sempre mantida; as reformulações aproximam a linguagem do gestor da do cliente."""
        if not self.use_query_rewriting:
            return None, [question]
        analysis = llm.analyze_question(question)
        queries = [question] + [q for q in analysis.search_queries if q.strip()][:4]
        return analysis, queries

    def ask(self, question: str, where: dict | None = None) -> RAGResponse:
        analysis, queries = self.transform_question(question)

        # Guarda 1: pergunta fora do escopo da base -> não recupera nem gera.
        if analysis is not None and not analysis.in_scope:
            return RAGResponse(
                question=question, status=STATUS_OUT_OF_SCOPE, analysis=analysis,
                answer=("Esta pergunta está fora do escopo da base de conhecimento, que contém apenas avaliações de "
                        f"clientes do e-commerce. Motivo: {analysis.reason}"),
            )

        # Desafio adicional: filtro explícito tem prioridade; senão, usa os filtros extraídos da pergunta.
        if where is None and self.use_metadata_filters and analysis is not None:
            where = build_where(analysis.filters)

        retrieval = self.retriever.retrieve(queries, where=where)

        # Guarda 2: poucas avaliações acima do limiar de similaridade -> não há base para responder.
        if len(retrieval.evidences) < config.MIN_EVIDENCES:
            return RAGResponse(
                question=question, status=STATUS_NO_EVIDENCE, analysis=analysis, retrieval=retrieval,
                evidences=retrieval.evidences, filters_applied=where,
                answer=("Não encontrei avaliações suficientemente relacionadas a esta pergunta para sustentar uma "
                        f"resposta ({len(retrieval.evidences)} avaliação(ões) acima do limiar de relevância; mínimo "
                        f"{config.MIN_EVIDENCES}). Prefiro não responder a gerar uma conclusão sem evidências."),
            )

        context = build_context(retrieval, where)
        generated = llm.generate_answer(question, context)

        # Guarda 3: validação das citações (rastreabilidade) e da suficiência declarada pelo modelo.
        valid = {e.ref for e in retrieval.evidences}
        found = extract_citations(generated.answer)
        cited = sorted((r for r in found if r in valid), key=lambda r: int(r[1:]))
        invalid = [r for r in found if r not in valid]

        status = STATUS_ANSWERED if generated.sufficient_evidence and cited else STATUS_INSUFFICIENT
        return RAGResponse(
            question=question, status=status, answer=generated.answer, evidences=retrieval.evidences,
            cited_refs=cited, invalid_refs=invalid, limitations=generated.limitations,
            analysis=analysis, retrieval=retrieval, context=context, model_used=llm.last_model_used,
            filters_applied=where,
        )
