"""Integração com o Gemini: análise/transformação da pergunta e geração da resposta fundamentada."""
import time
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field

from . import config
from .filters import MetadataFilters, categories


class QueryAnalysis(BaseModel):
    question_type: Literal["experiencia_do_cliente", "dados_internos_da_empresa", "fora_do_dominio"] = Field(
        description="Tipo da pergunta, decidido apenas pelo assunto (não pela existência de dados)"
    )
    reason: str = Field(description="Justificativa curta da classificação")
    search_queries: list[str] = Field(
        description="2 a 4 frases de busca escritas como um cliente escreveria uma avaliação sobre o tema"
    )
    @property
    def in_scope(self) -> bool:
        # A existência de avaliações sobre o tema é verificada pela busca, não pelo LLM.
        return self.question_type == "experiencia_do_cliente"

    filters: MetadataFilters = Field(
        default_factory=MetadataFilters,
        description="Filtros de metadados SOMENTE quando a pergunta pede explicitamente um recorte",
    )


class GroundedAnswer(BaseModel):
    sufficient_evidence: bool = Field(description="As avaliações fornecidas permitem responder à pergunta?")
    answer: str = Field(description="Resposta em markdown, com citações [R#] após cada afirmação")
    limitations: str = Field(description="Limitações da resposta (tamanho da amostra, vieses, o que não foi possível concluir)")


ANALYZE_PROMPT = """Você é o módulo de processamento de perguntas de um sistema de Voice of Customer.
A base de conhecimento contém APENAS avaliações escritas por clientes de um marketplace brasileiro (Olist, 2016-2018):
comentários sobre produtos, entrega, prazos, atendimento, vendedores, embalagem, preço etc.

Tarefas:
1. Classifique a pergunta (question_type) SOMENTE pelo assunto:
   - "experiencia_do_cliente": qualquer aspecto que um cliente poderia comentar numa avaliação (produto, entrega,
     atendimento, pagamento, formas de compra, preço, site, vendedor...), MESMO que você ache que o assunto é raro,
     inexistente ou anacrônico na base. Você NÃO sabe o que há nas avaliações; isso será verificado pela busca.
   - "dados_internos_da_empresa": métricas que avaliações não contêm (faturamento, lucro, estoque, nº de funcionários).
   - "fora_do_dominio": sem relação com compras ou com a experiência de clientes (geografia, receitas, política...).
2. Reescreva a pergunta em 2 a 4 consultas de busca. As avaliações são textos curtos e informais em primeira pessoa,
   então escreva as consultas como trechos que um cliente escreveria (ex.: pergunta "principais problemas de entrega"
   -> "o produto não chegou até hoje", "a entrega atrasou muito além do prazo"). Cubra aspectos diferentes do tema.
   As consultas devem descrever o conteúdo do comentário, não o recorte (nota, categoria, data vão nos filtros).
3. Extraia filtros de metadados (filters) APENAS se a pergunta pedir explicitamente um recorte; caso contrário,
   deixe todos os campos nulos. Recortes possíveis:
   - nota: "nota baixa"/"clientes insatisfeitos" -> max_score=2; "nota alta"/"clientes satisfeitos" -> min_score=4;
   - categoria de produto (use somente nomes desta lista): {categories};
   - período (AAAA-MM; a base vai de 2016-09 a 2018-10), ex.: "maio de 2018" -> month_from=month_to="2018-05";
   - entrega após o prazo estimado (delivered_late), ex.: "pedidos entregues com atraso" -> true;
   - UF do cliente (customer_states), ex.: "clientes do Rio de Janeiro" -> ["RJ"].
   Perguntas gerais sobre um tema (ex.: "problemas de entrega") NÃO devem gerar filtros: o tema é tratado pela busca.

Pergunta do usuário: {question}"""

ANSWER_SYSTEM = """Você é um analista de Voice of Customer. Responda à pergunta do gestor usando EXCLUSIVAMENTE as
avaliações de clientes fornecidas no contexto.

Regras obrigatórias:
- Toda afirmação deve ser sustentada por avaliações e citada no formato [R1], [R2, R5]. Não cite IDs inexistentes.
- Não use conhecimento externo e não invente fatos, causas, números ou percentuais.
- Para falar de frequência ou recorrência, use apenas o "Perfil das avaliações relacionadas" e o número de avaliações
  que sustentam cada ponto; deixe claro que é uma estimativa a partir da amostra recuperada.
- Agrupe os achados em temas/insights (bullets com um título curto em negrito), do mais para o menos sustentado.
- Se as avaliações não forem suficientes ou não tratarem do que foi perguntado, marque sufficient_evidence = false
  e explique, em answer, o que foi e o que não foi encontrado.
- Escreva em português do Brasil, de forma objetiva."""


@lru_cache(maxsize=1)
def get_client():
    from google import genai
    from google.genai import types

    if not config.GEMINI_API_KEY:
        raise RuntimeError("Defina GEMINI_API_KEY no arquivo .env (veja .env.example).")
    return genai.Client(api_key=config.GEMINI_API_KEY, http_options=types.HttpOptions(timeout=90_000))


_TRANSIENT_CODES = {500, 503, 504}
last_model_used: str | None = None  # modelo que respondeu a última chamada (principal ou reserva)
_cooldown_until: dict[str, float] = {}


def _generate(prompt: str, schema: type[BaseModel], system: str | None = None) -> BaseModel:
    """Chamada com saída estruturada e resiliência para o plano gratuito:
    - 429 (cota esgotada): o modelo é pulado por LLM_COOLDOWN_SECONDS e o próximo da lista é usado;
    - 5xx (sobrecarga): nova tentativa curta e, se persistir, o próximo modelo."""
    from google.genai import errors, types

    global last_model_used
    gen_config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=config.TEMPERATURE,
        response_mime_type="application/json",
        response_schema=schema,
    )
    models = [config.GEMINI_MODEL, *[m for m in config.GEMINI_FALLBACK_MODELS if m != config.GEMINI_MODEL]]
    available = [m for m in models if _cooldown_until.get(m, 0) < time.time()] or models
    last_error = None
    for model in available:
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                resp = get_client().models.generate_content(model=model, contents=prompt, config=gen_config)
                last_model_used = model
                return resp.parsed if resp.parsed is not None else schema.model_validate_json(resp.text)
            except errors.APIError as e:
                last_error = e
                if e.code == 429:
                    _cooldown_until[model] = time.time() + config.LLM_COOLDOWN_SECONDS
                    break
                if e.code not in _TRANSIENT_CODES:
                    raise
                if attempt < config.LLM_MAX_RETRIES - 1:
                    time.sleep(3)
    raise RuntimeError(f"Gemini indisponível em todos os modelos ({', '.join(available)}): {last_error}")


def analyze_question(question: str) -> QueryAnalysis:
    return _generate(ANALYZE_PROMPT.format(question=question, categories=", ".join(categories())), QueryAnalysis)


def generate_answer(question: str, context: str) -> GroundedAnswer:
    prompt = f"{context}\n\n# Pergunta do gestor\n{question}"
    return _generate(prompt, GroundedAnswer, system=ANSWER_SYSTEM)
