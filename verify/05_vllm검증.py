# =====================================================================
# [VESSL 점검 5] vLLM 계열 노트북 3종 검증
#
#   전제: verify/02_vllm_venv.sh 로 서버가 떠 있어야 합니다.
#         curl -s http://localhost:8000/v1/models  로 확인하세요.
#
#   ※ 이 셀은 **기본 커널**에서 실행합니다 (vllm venv 아님).
#      HTTP 로 붙기 때문에 openai / llama-index 만 있으면 됩니다.
#
#   붙여넣어 한 번에 실행. 5~8분. 출력 전체를 회신해 주세요.
#
# 확인 대상 — work/notebook/ 수정본에 적용한 변경들
#   A. 서버 기동 / Qwen3-4B 로딩          (퓨샷·Amazon·RAG 공통 전제)
#   B. guided_choice 제거 → structured_outputs  (퓨샷 3곳)
#   C. guided_json 제거 → response_format       (퓨샷 2곳 · Amazon 6곳)
#   D. ★ BM25 RAG 프롬프트가 실제로 반영되는가   (D-3 — 가장 중요)
#   E. Settings.embed_model 없이 BM25 인덱싱이 되는가
# =====================================================================
import json
import time

BASE = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
R = {}


def hr(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def probe(key, fn):
    try:
        out = fn()
        R[key] = {"ok": True, "detail": str(out)[:400]}
        print(f"  [OK]   {key}: {str(out)[:280]}")
        return out
    except Exception as e:  # noqa: BLE001
        R[key] = {"ok": False, "detail": f"{type(e).__name__}: {e}"[:600]}
        print(f"  [FAIL] {key}")
        print("         " + f"{type(e).__name__}: {e}"[:600].replace("\n", "\n         "))
        return None


# ---------------------------------------------------------------------
hr("A. 서버 기동 및 모델 확인")
# ---------------------------------------------------------------------
from openai import OpenAI  # noqa: E402

client = OpenAI(api_key="EMPTY", base_url=BASE)

served = probe("served_models", lambda: [m.id for m in client.models.list().data])
if not served:
    raise SystemExit(
        "\n★ 서버에 붙지 못했습니다. 02_vllm_venv.sh 로 서버를 먼저 띄우세요.\n"
        "   tail -n 100 /tmp/vllm.log 로 로그를 확인하고 회신해 주세요."
    )

probe("basic_chat", lambda: client.chat.completions.create(
    model=MODEL, messages=[{"role": "user", "content": "한국의 수도는?"}],
    max_tokens=32).choices[0].message.content)

# ---------------------------------------------------------------------
hr("B. guided_choice — 제거됐는지 확인 (퓨샷 셀 23·26·43)")
# ---------------------------------------------------------------------
MSG = [{"role": "system", "content": "You are a helpful assistant."},
       {"role": "user", "content": "감정 분석: 이 영화 정말 재밌었어요"}]

print("① 구 방식 — 실패해야 정상입니다")
probe("guided_choice_구방식", lambda: client.chat.completions.create(
    model=MODEL, messages=MSG,
    extra_body={"guided_choice": ["긍정", "부정"]}).choices[0].message.content)

print("\n② 수정본이 쓰는 방식")
probe("structured_outputs_신방식", lambda: client.chat.completions.create(
    model=MODEL, messages=MSG,
    extra_body={"structured_outputs": {"choice": ["긍정", "부정"]}}
).choices[0].message.content)

# ---------------------------------------------------------------------
hr("C. guided_json → response_format (퓨샷 2곳 · Amazon 6곳)")
# ---------------------------------------------------------------------
from pydantic import BaseModel  # noqa: E402


class NameDescription(BaseModel):
    names: list[str]


schema = NameDescription.model_json_schema()
QMSG = [{"role": "user", "content": "다음에서 인물 이름만 나열: 홍길동과 김철수가 서울에서 만났다"}]

print("① 구 방식 — 실패해야 정상입니다")
probe("guided_json_구방식", lambda: client.chat.completions.create(
    model=MODEL, messages=QMSG,
    extra_body={"guided_json": schema}).choices[0].message.content)

print("\n② 수정본이 쓰는 방식 (OpenAI 표준 response_format)")
out = probe("response_format_신방식", lambda: client.chat.completions.create(
    model=MODEL, messages=QMSG,
    response_format={"type": "json_schema",
                     "json_schema": {"name": "answer", "schema": schema}},
).choices[0].message.content)

if out:
    probe("response_format_json파싱", lambda: json.loads(out))

print("\n③ Amazon 노트북이 쓰는 중첩 스키마도 되는지")


class FeatureType(BaseModel):
    feature_type: str
    descript: str


class FeatureTypeList(BaseModel):
    feature_type_list: list[FeatureType]


nested = FeatureTypeList.model_json_schema()
o2 = probe("response_format_중첩스키마", lambda: client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content":
               "이 제품의 특징 2가지를 뽑아줘: 무선 블루투스 이어폰, 방수, 30시간 재생"}],
    temperature=0.8, top_p=0.95,
    response_format={"type": "json_schema",
                     "json_schema": {"name": "feature_type_list", "schema": nested}},
).choices[0].message.content)
if o2:
    probe("중첩스키마_json파싱", lambda: json.loads(o2))

# ---------------------------------------------------------------------
hr("D. ★ BM25 RAG 프롬프트 반영 — 이번 검증의 핵심")
# ---------------------------------------------------------------------
print("""수정 전 코드는 get_prompts() 가 돌려준 객체의 template 을 직접 고쳤습니다.
get_prompts() 는 deepcopy 를 반환하므로 엔진에 반영되지 않는데, 에러가 안 나고
다시 출력하면 바뀐 것처럼 보여서 수강생이 적용된 줄 알고 넘어갑니다.
여기서 그 사실을 직접 확인하고, update_prompts() 가 실제로 반영되는지 봅니다.
""")


def rag_prompt_test():
    from llama_index.core import Document, PromptTemplate, Settings, get_response_synthesizer
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.llms.openai_like import OpenAILike
    from llama_index.retrievers.bm25 import BM25Retriever
    import Stemmer

    Settings.llm = OpenAILike(model=MODEL, api_base=BASE, api_key="EMPTY",
                              is_chat_model=True)
    Settings.embed_model = None      # BM25 는 어휘 기반이라 임베딩 불필요

    docs = [Document(text=t) for t in [
        "백남준은 대한민국의 비디오 아티스트이다. 1932년 서울에서 태어났다.",
        "맥스웰 방정식은 전자기 현상을 기술하는 네 개의 편미분 방정식이다.",
        "세종대왕은 조선의 4대 임금으로 훈민정음을 창제하였다.",
    ]]
    nodes = SentenceSplitter(chunk_size=256).get_nodes_from_documents(docs)

    retr = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2,
                                       stemmer=Stemmer.Stemmer("english"),
                                       language="english")
    qe = RetrieverQueryEngine(
        retriever=retr,
        response_synthesizer=get_response_synthesizer(response_mode="refine"))

    KEY = "response_synthesizer:refine_template"
    before = qe.get_prompts()[KEY].get_template()

    # --- 구 방식: 반환된 사본을 직접 수정 ---
    p = qe.get_prompts()
    MARK_OLD = "★구방식마커★"
    try:
        p[KEY].default_template.template = MARK_OLD + " {query_str} {existing_answer} {context_msg}"
    except Exception as e:  # noqa: BLE001
        print(f"    (구 방식은 대입 자체가 실패: {type(e).__name__})")
    after_old = qe.get_prompts()[KEY].get_template()
    old_applied = MARK_OLD in after_old

    # --- 신 방식: update_prompts() ---
    MARK_NEW = "★신방식마커★"
    qe.update_prompts({KEY: PromptTemplate(
        MARK_NEW + " {query_str} {existing_answer} {context_msg}")})
    after_new = qe.get_prompts()[KEY].get_template()
    new_applied = MARK_NEW in after_new

    return {
        "구방식_반영됨": old_applied,
        "신방식_반영됨": new_applied,
        "원본_앞부분": before[:60],
    }


res = probe("rag_prompt_반영여부", rag_prompt_test)
if res and isinstance(res, dict):
    print()
    if res["구방식_반영됨"] is False and res["신방식_반영됨"] is True:
        print("  ★ 확인: 구 방식은 무시되고 update_prompts() 만 반영됩니다.")
        print("    → D-3 지적이 실측으로 확정됐고, 수정본이 올바릅니다.")
    elif res["구방식_반영됨"]:
        print("  ※ 구 방식도 반영됐습니다. D-3 지적을 재검토해야 합니다.")
    if not res["신방식_반영됨"]:
        print("  ★ update_prompts() 가 반영되지 않았습니다. 수정본을 다시 봐야 합니다.")

# ---------------------------------------------------------------------
hr("E. BM25 + embed_model=None 으로 실제 질의")
# ---------------------------------------------------------------------
def rag_query():
    from llama_index.core import (Document, PromptTemplate, Settings,
                                  get_response_synthesizer)
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.retrievers.bm25 import BM25Retriever
    import Stemmer

    docs = [Document(text=t) for t in [
        "백남준은 대한민국의 비디오 아티스트이다. 1932년 서울에서 태어나 비디오 아트의 창시자로 불린다.",
        "맥스웰 방정식은 전자기 현상을 기술하는 네 개의 편미분 방정식이다.",
    ]]
    nodes = SentenceSplitter(chunk_size=256).get_nodes_from_documents(docs)
    retr = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2,
                                       stemmer=Stemmer.Stemmer("english"),
                                       language="english")
    qe = RetrieverQueryEngine(
        retriever=retr,
        response_synthesizer=get_response_synthesizer(response_mode="refine"))
    qe.update_prompts({"response_synthesizer:refine_template": PromptTemplate(
        "원 질의: {query_str}\n기존 응답: {existing_answer}\n"
        "새 컨텍스트:\n{context_msg}\n"
        "새 컨텍스트를 참고해 더 나은 답을 **한국어로** 주세요.\n개선된 답변: ")})

    t0 = time.time()
    resp = qe.query("백남준은 누구인가요?")
    return {"소요초": round(time.time() - t0, 1),
            "응답": str(resp)[:300],
            "출처노드수": len(resp.source_nodes)}


probe("rag_query_실행", rag_query)

# ---------------------------------------------------------------------
hr("결과 요약")
# ---------------------------------------------------------------------
print("기대하는 결과:")
print("  · guided_choice_구방식 / guided_json_구방식 → [FAIL]  (제거됐으므로 정상)")
print("  · structured_outputs_신방식 / response_format_신방식 → [OK]")
print("  · rag_prompt_반영여부 → 구방식 False / 신방식 True")
print()
fails = [k for k, v in R.items() if isinstance(v, dict) and not v.get("ok")]
print(f"실패 {len(fails)}건: {fails}")
print("\n--- 아래 JSON 도 함께 회신해 주세요 ---")
print(json.dumps(R, ensure_ascii=False, indent=1)[:9000])
