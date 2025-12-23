from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import neo4j
from dotenv import load_dotenv
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever, VectorCypherRetriever, Text2CypherRetriever, ToolsRetriever
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.generation import RagTemplate, GraphRAG

# Load environment variables
load_dotenv()

# Configuration
URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD", "12345678"))
INDEX_NAME = "content_vector_index"

app = FastAPI(title="RAG Search API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
driver = None
rag = None

# Request/Response Models
class QueryRequest(BaseModel):
    query: str

class Source(BaseModel):
    id: int
    shortName: str
    title: str
    category: str
    date: str
    url: str
    summary: str
    icon: str

class Section(BaseModel):
    title: str
    content: str
    sourceIds: List[int]

class QueryResponse(BaseModel):
    sections: List[Section]
    sources: List[Source]

def get_schema(driver):
    """Neo4j 데이터베이스의 스키마 정보를 가져옵니다"""
    with driver.session() as session:
        node_info = session.run("""
            CALL db.schema.nodeTypeProperties()
            YIELD nodeType, propertyName, propertyTypes
            RETURN nodeType, collect(propertyName) as properties
        """).data()

        patterns = session.run("""
            MATCH (n)-[r]->(m)
            RETURN DISTINCT labels(n)[0] as source, type(r) as relationship, labels(m)[0] as target
            LIMIT 20
        """).data()

        schema_text = "=== Neo4j Schema ===\n"
        schema_text += "\n노드 타입:\n"
        for node in node_info:
            schema_text += f"- {node['nodeType']}: {node['properties']}\n"

        schema_text += "\n관계 패턴:\n"
        for pattern in patterns:
            schema_text += f"- ({pattern['source']})-[:{pattern['relationship']}]->({pattern['target']})\n"

        return schema_text

def initialize_graphrag():
    """GraphRAG 시스템 초기화"""
    global driver, rag
    
    try:
        driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)
        driver.verify_connectivity()
        print("✓ Neo4j 연결 성공")
    except Exception as e:
        print(f"✗ Neo4j 연결 실패: {e}")
        return False

    llm = OpenAILLM(
        model_name="gpt-4o",
        model_params={"temperature": 0}
    )
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 벡터 임베딩 생성 (없는 경우)
    print("벡터 임베딩 확인 중...")
    from neo4j_graphrag.indexes import create_vector_index
    
    with driver.session() as session:
        # 임베딩 없는 Content 노드 확인
        result = session.run("MATCH (c:Content) WHERE c.embedding IS NULL RETURN elementId(c) AS id, c.chunk AS text")
        records = result.data()
        
        if records:
            print(f"  → {len(records)}개 청크에 임베딩 생성 중...")
            for i, record in enumerate(records):
                node_id = record["id"]
                text = record["text"]
                try:
                    vector = embedder.embed_query(text)
                    if hasattr(vector, 'tolist'):
                        vector = vector.tolist()
                    
                    session.run("""
                        MATCH (c) WHERE elementId(c) = $id
                        SET c.embedding = $embedding
                        """, {"id": node_id, "embedding": vector})
                    
                    if (i+1) % 10 == 0:
                        print(f"  → 처리됨: {i+1}/{len(records)}")
                except Exception as e:
                    print(f"  ✗ 청크 {node_id} 임베딩 오류: {e}")
            print("✓ 임베딩 생성 완료")
        else:
            print("✓ 모든 청크에 임베딩 존재")
    
    # 벡터 인덱스 생성
    try:
        create_vector_index(
            driver,
            INDEX_NAME,
            label="Content",
            embedding_property="embedding",
            dimensions=1536,
            similarity_fn="cosine",
        )
        print("✓ 벡터 인덱스 생성/확인 완료")
    except Exception as e:
        print(f"  ℹ 인덱스 정보: {e}")

    # Vector Retriever (결과 개수 증가)
    vector_retriever = VectorRetriever(
        driver=driver,
        index_name=INDEX_NAME,
        embedder=embedder
    )
    
    # VectorCypher Retriever
    retrieval_query = """
    WITH node AS content, score
    MATCH (content)<-[:HAS_CHUNK]-(article:Article)
    OPTIONAL MATCH (article)-[:BELONGS_TO]->(category:Category)
    OPTIONAL MATCH (media:Media)-[:PUBLISHED]->(article)
    OPTIONAL MATCH (category)<-[:BELONGS_TO]-(related_article:Article)
    WHERE related_article <> article

    RETURN
        content.content_id AS content_id,
        content.chunk AS chunk,
        content.title AS content_title,
        article.article_id AS article_id,
        article.title AS article_title,
        article.url AS article_url,
        article.published_date AS article_date,
        category.name AS category_name,
        media.name AS media_name,
        score AS similarity_score,
        collect(DISTINCT {
            article_id: related_article.article_id,
            title: related_article.title,
            url: related_article.url,
            published_date: related_article.published_date
        })[0..5] AS related_articles
    """
    
    vector_cypher_retriever = VectorCypherRetriever(
        driver=driver,
        index_name=INDEX_NAME,
        retrieval_query=retrieval_query,
        embedder=embedder
    )

    # Text2Cypher Retriever
    neo4j_schema = get_schema(driver)
    
    examples = [
        """
        USER INPUT: 경제 분야의 최신 뉴스 알려주세요
        CYPHER QUERY:
        MATCH (a:Article)-[:BELONGS_TO]->(c:Category {name: "경제"})
        RETURN a.article_id, a.title, a.url, a.published_date
        ORDER BY a.published_date DESC
        LIMIT 10
        """,
        """
        USER INPUT: 매일경제에서 나온 최신 뉴스 3개 보여주세요
        CYPHER QUERY:
        MATCH (m:Media {name: "매일경제"})-[:PUBLISHED]->(a:Article)
        RETURN a.article_id, a.title, a.url, a.published_date
        ORDER BY a.published_date DESC
        LIMIT 3
        """,
        """
        USER INPUT: 카테고리별 기사 개수를 알려주세요
        CYPHER QUERY:
        MATCH (a:Article)-[:BELONGS_TO]->(c:Category)
        RETURN c.name as category, count(a) as article_count
        ORDER BY article_count DESC
        """,
    ]
    
    text2cypher_retriever = Text2CypherRetriever(
        driver=driver,
        llm=llm,
        neo4j_schema=neo4j_schema,
        examples=examples,
    )

    # Tools Setup
    vector_tool = vector_retriever.convert_to_tool(
        name="vector_retriever",
        description="벡터 기반 검색으로 뉴스기사에 등장하는 내용 텍스트를 기반으로 검색할 때 사용합니다."
    )
    vector_cypher_tool = vector_cypher_retriever.convert_to_tool(
        name="vectorcypher_retriever",
        description="벡터 검색으로 찾아진 Content와 연결된 Article을 기준으로, 그 기사의 상세한 정보는 물론 같은 카테고리의 다른 기사들을 함께 반환합니다."
    )
    text2cypher_tool = text2cypher_retriever.convert_to_tool(
        name="text2cypher_retriever",
        description="text2cypher 검색 기반으로 언론사, 분야별 기사 등 엔티티 혹은 속성을 기반으로 정보를 찾을 때 사용합니다."
    )

    tools_retriever = ToolsRetriever(
        driver=driver,
        llm=llm,
        tools=[vector_tool, vector_cypher_tool, text2cypher_tool],
    )

    # GraphRAG Setup
    prompt_template = RagTemplate(
        template="""당신은 뉴스 기사 정보를 제공하는 전문 어시스턴트입니다.

질문: {query_text}

검색된 문서 정보:
{context}

지침:
1. 제공된 검색 결과에서 **최소 10개 이상**의 뉴스 기사를 선택하여 답변하세요.
2. **섹션을 나누지 말고** 모든 뉴스를 하나의 리스트로 제공하세요.
3. 각 뉴스마다 제목, URL, 발행일, 카테고리, 언론사, 요약(2-3문장)을 반드시 포함하세요.
4. 검색 결과에 없는 내용은 추측하지 마세요.
5. 다음 JSON 형식으로만 답변하세요 (마크다운 코드 블록 없이):

{{
  "sections": [
    {{
      "title": "검색 결과",
      "content": "",
      "sources": [
        {{
          "title": "기사 제목",
          "url": "기사 URL",
          "date": "발행일",
          "category": "카테고리",
          "media": "언론사",
          "summary": "기사 요약 (2-3문장)"
        }}
      ]
    }}
  ]
}}

답변:""",
        expected_inputs=["context", "query_text"]
    )

    rag = GraphRAG(
        llm=llm,
        retriever=tools_retriever,
        prompt_template=prompt_template
    )
    
    print("✓ GraphRAG 시스템 초기화 완료")
    return True

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 GraphRAG 초기화"""
    success = initialize_graphrag()
    if not success:
        print("⚠ Warning: GraphRAG 초기화 실패")

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 연결 해제"""
    global driver
    if driver:
        driver.close()
        print("✓ Neo4j 연결 종료")

@app.get("/")
async def root():
    return {"message": "RAG Search API is running"}

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    global driver
    try:
        if driver:
            driver.verify_connectivity()
            return {"status": "healthy", "database": "connected"}
        return {"status": "unhealthy", "database": "not initialized"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/search", response_model=QueryResponse)
async def search(request: QueryRequest):
    """검색 쿼리 처리"""
    global rag
    
    if not rag:
        raise HTTPException(status_code=503, detail="GraphRAG system not initialized")
    
    try:
        # GraphRAG 검색 실행
        result = rag.search(query_text=request.query, return_context=True)
        
        # 응답에서 마크다운 코드 블록 제거
        answer_text = result.answer.strip()
        
        # ```json ... ``` 형식 제거
        if answer_text.startswith('```'):
            # 첫 줄 제거 (```json)
            lines = answer_text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            # 마지막 줄 제거 (```)
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            answer_text = '\n'.join(lines).strip()
        
        # JSON 파싱
        import json
        try:
            parsed_result = json.loads(answer_text)
        except Exception as e:
            print(f"JSON 파싱 실패: {e}")
            print(f"응답 내용: {answer_text[:500]}")
            # JSON 파싱 실패 시 기본 형태로 반환
            parsed_result = {
                "sections": [{
                    "title": "검색 결과",
                    "content": result.answer,
                    "sources": []
                }]
            }
        
        # 출처 정보 변환
        sources = []
        source_id = 1
        
        for section in parsed_result.get("sections", []):
            source_ids = []
            for source_data in section.get("sources", []):
                sources.append({
                    "id": source_id,
                    "shortName": source_data.get("media", "unknown"),
                    "title": source_data.get("title", ""),
                    "category": source_data.get("category", "기타"),
                    "date": source_data.get("date", ""),
                    "url": source_data.get("url", ""),
                    "summary": source_data.get("summary", ""),
                    "icon": get_icon_for_category(source_data.get("category", ""))
                })
                source_ids.append(source_id)
                source_id += 1
            
            section["sourceIds"] = source_ids
            # sources 키 제거 (프론트엔드에서 sourceIds 사용)
            section.pop("sources", None)
        
        return {
            "sections": parsed_result.get("sections", []),
            "sources": sources
        }
        
    except Exception as e:
        print(f"검색 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

def get_icon_for_category(category: str) -> str:
    """카테고리에 따른 아이콘 반환"""
    icons = {
        "정치": "🏛️",
        "경제": "💼",
        "사회": "👥",
        "생활/문화": "🎭",
        "IT/과학": "💻",
        "세계": "🌍",
    }
    return icons.get(category, "📰")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)