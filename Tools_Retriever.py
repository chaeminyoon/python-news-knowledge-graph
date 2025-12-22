import os
import sys
import neo4j
from dotenv import load_dotenv
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever, VectorCypherRetriever, Text2CypherRetriever, ToolsRetriever
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.generation import RagTemplate, GraphRAG

# Load environment variables
load_dotenv()

# Configuration
URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD", "12345678"))
INDEX_NAME = "content_vector_index"
DIMENSION = 1536

def get_schema(driver):
    """Neo4j 데이터베이스의 스키마 정보를 가져옵니다"""
    with driver.session() as session:
        # 노드 라벨과 속성 정보
        node_info = session.run("""
            CALL db.schema.nodeTypeProperties()
            YIELD nodeType, propertyName, propertyTypes
            RETURN nodeType, collect(propertyName) as properties
        """).data()

        # 관계 정보
        # rel_info query omitted as not strictly used in text formation logic in notebook but patterns are
        
        # 관계 패턴 정보
        patterns = session.run("""
            MATCH (n)-[r]->(m)
            RETURN DISTINCT labels(n)[0] as source, type(r) as relationship, labels(m)[0] as target
            LIMIT 20
        """).data()

        schema_text = "=== Neo4j Schema ===\n"

        # 노드 정보 추가
        schema_text += "\n노드 타입:\n"
        for node in node_info:
            schema_text += f"- {node['nodeType']}: {node['properties']}\n"

        # 관계 정보 추가
        schema_text += "\n관계 패턴:\n"
        for pattern in patterns:
            schema_text += f"- ({pattern['source']})-[:{pattern['relationship']}]->({pattern['target']})\n"

        return schema_text

def initialize_graphrag():
    print("Initializing GraphRAG System...")
    
    # 1. Driver & LLM Setup
    try:
        driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)
        driver.verify_connectivity()
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return None, None

    llm = OpenAILLM(
        model_name="gpt-4o",
        model_params={"temperature": 0}
    )
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    # 2. Vector Indexing Setup
    print("Checking/Updating Vector Embeddings...")
    with driver.session() as session:
        # Check for chunks without embeddings
        result = session.run("MATCH (c:Content) WHERE c.embedding IS NULL RETURN elementId(c) AS id, c.chunk AS text")
        records = result.data()
        
        if records:
            print(f"Found {len(records)} chunks without embeddings. Generating...")
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
                        print(f"Processed {i+1}/{len(records)}")
                except Exception as e:
                    print(f"Error embedding chunk {node_id}: {e}")
        else:
            print("All chunks have embeddings.")

    # Create Index
    try:
        create_vector_index(
            driver,
            INDEX_NAME,
            label="Content",
            embedding_property="embedding",
            dimensions=DIMENSION,
            similarity_fn="cosine",
        )
        print("Vector index ensured.")
    except Exception as e:
        # Index might already exist
        print(f"Index creation note: {e}")

    # 3. Define Retrievers
    
    # A. Vector Retriever
    vector_retriever = VectorRetriever(
        driver=driver,
        index_name=INDEX_NAME,
        embedder=embedder
    )
    
    # B. VectorCypher Retriever
    retrieval_query = """
    WITH node AS content, score
    MATCH (content)<-[:HAS_CHUNK]-(article:Article)
    OPTIONAL MATCH (article)-[:BELONGS_TO]->(category:Category)
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

    # C. Text2Cypher Retriever
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
        USER INPUT: 2025년 11월 1일 이후에 발행된 정치 관련 기사는 몇 개나 되나요?
        CYPHER QUERY:
        MATCH (a:Article)-[:BELONGS_TO]->(c:Category {name: "정치"})
        WHERE a.published_date >= "2025-11-01"
        RETURN count(a) as article_count
        """,
        """
        USER INPUT: 카테고리별 기사 개수를 알려주세요
        CYPHER QUERY:
        MATCH (a:Article)-[:BELONGS_TO]->(c:Category)
        RETURN c.name as category, count(a) as article_count
        ORDER BY article_count DESC
        """,
        """
        USER INPUT: 11월 2일에 발행된 기사 중 정치 분야는?
        CYPHER QUERY:
        MATCH (a:Article)-[:BELONGS_TO]->(c:Category {name: "정치"})
        WHERE a.published_date STARTS WITH "2025-11-02"
        RETURN a.article_id, a.title, a.url, a.published_date
        ORDER BY a.published_date DESC
        """,
    ]
    
    text2cypher_retriever = Text2CypherRetriever(
        driver=driver,
        llm=llm,
        neo4j_schema=neo4j_schema,
        examples=examples,
    )

    # 4. Tools Setup
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

    # 5. GraphRAG Setup
    prompt_template = RagTemplate(
        template="""당신은 뉴스 기사 정보를 제공하는 전문 어시스턴트입니다.

질문: {query_text}

검색된 문서 정보:
{context}

지침:
1. 제공된 검색 결과(Context)의 내용을 충실히 사용하여 답변하세요.
2. 답변에는 반드시 관련 뉴스 기사의 **제목(title)**과 **URL(url)**을 포함해야 합니다.
3. 여러 기사가 검색된 경우, 각 기사의 출처를 명확히 구분하여 제시하세요.
4. 검색 결과에 없는 내용은 추측하지 마세요.
5. 가능하면 발행일(published_date)과 카테고리(category_name)도 함께 언급하세요.

답변 형식 예시:
- [기사 제목] (카테고리명, 발행일)
  URL: [기사 URL]
  내용: [관련 내용 요약]

답변:""",
        expected_inputs=["context", "query_text"]
    )

    rag = GraphRAG(
        llm=llm,
        retriever=tools_retriever,
        prompt_template=prompt_template
    )
    
    return rag, driver

def main():
    rag, driver = initialize_graphrag()
    
    if not rag:
        print("System initialization failed.")
        return

    print("\n" + "="*50)
    print(" 뉴스 기사 AI 검색 시스템 (Exit: 'q' or 'quit')")
    print("="*50 + "\n")

    try:
        while True:
            query = input("질문을 입력하세요: ").strip()
            
            if query.lower() in ('q', 'quit', 'exit'):
                break
            
            if not query:
                continue

            print("\n검색 중...\n")
            
            try:
                result = rag.search(query_text=query, return_context=True)
                
                print("============== [ 답변 결과 ] ==============")
                print(result.answer)
                print("==========================================\n")
                
                # Context Debug (Optional)
                if hasattr(result, 'retriever_result') and result.retriever_result:
                    print(f"[참고: 검색된 소스 {len(result.retriever_result.items)}개]\n")
                    # un-comment to see details
                    # for idx, item in enumerate(result.retriever_result.items, 1):
                    #     print(f"Source {idx} Metadata: {item.metadata}")
                    
            except Exception as e:
                print(f"Error processing query: {e}")

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if driver:
            driver.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()
