import pandas as pd
import neo4j
import os
import glob
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neo4j Connection Configuration
URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD", "12345678"))

def chunk_text(text, chunk_size=500, overlap=50):
    if pd.isna(text) or text == '':
        return []

    text = str(text)
    chunks = []

    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks

def clear_database(tx):
    """데이터베이스의 모든 노드와 관계를 삭제합니다"""
    tx.run("MATCH (n) DETACH DELETE n")

def create_constraints(tx):
    """유니크 제약조건을 생성합니다"""
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.article_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Content) REQUIRE c.content_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Media) REQUIRE m.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (cat:Category) REQUIRE cat.name IS UNIQUE"
    ]

    for constraint in constraints:
        try:
            tx.run(constraint)
        except Exception as e:
            print(f"제약조건 생성 중 오류: {e}")

def create_article_node(tx, article_data):
    query = """
    MERGE (a:Article {article_id: $article_id})
    SET a.title = $title,
        a.url = $url,
        a.published_date = $published_date
    RETURN a
    """
    tx.run(query,
           article_id=article_data['article_id'],
           title=article_data['title'],
           url=article_data['url'],
           published_date=article_data['published_date'])

def create_content_nodes(tx, article_id, content_chunks, article_data):
    for i, chunk in enumerate(content_chunks):
        content_id = f"{article_id}_chunk_{i}"

        query = """
        MERGE (c:Content {content_id: $content_id})
        SET c.chunk = $chunk,
            c.article_id = $article_id,
            c.title = $title,
            c.url = $url,
            c.published_date = $published_date,
            c.chunk_index = $chunk_index
        """
        tx.run(query,
               content_id=content_id,
               chunk=chunk,
               article_id=article_id,
               title=article_data['title'],
               url=article_data['url'],
               published_date=article_data['published_date'],
               chunk_index=i)

        relationship_query = """
        MATCH (a:Article {article_id: $article_id})
        MATCH (c:Content {content_id: $content_id})
        MERGE (a)-[:HAS_CHUNK]->(c)
        """
        tx.run(relationship_query,
               article_id=article_id,
               content_id=content_id)

def create_media_node_and_relationship(tx, article_id, source):
    if pd.isna(source) or source == '':
        return

    media_query = """
    MERGE (m:Media {name: $source})
    RETURN m
    """
    tx.run(media_query, source=source)

    relationship_query = """
    MATCH (a:Article {article_id: $article_id})
    MATCH (m:Media {name: $source})
    MERGE (m)-[:PUBLISHED]->(a)
    """
    tx.run(relationship_query,
           article_id=article_id,
           source=source)

def create_category_node_and_relationship(tx, article_id, category):
    if pd.isna(category) or category == '':
        return

    category_query = """
    MERGE (cat:Category {name: $category})
    RETURN cat
    """
    tx.run(category_query, category=category)

    relationship_query = """
    MATCH (a:Article {article_id: $article_id})
    MATCH (cat:Category {name: $category})
    MERGE (a)-[:BELONGS_TO]->(cat)
    """
    tx.run(relationship_query,
           article_id=article_id,
           category=category)

def build_graph_from_dataframe(driver, df, chunk_size=500, overlap=50):
    with driver.session() as session:
        for idx, row in df.iterrows():
            try:
                article_id = row.get('article_id', '')

                article_data = {
                    'article_id': article_id,
                    'title': row.get('title', ''),
                    'url': row.get('url', ''),
                    'published_date': str(row.get('published_date', ''))
                }

                # 1. Article 노드 생성
                session.execute_write(create_article_node, article_data)

                # 2. Content 노드들 생성 (content 컬럼이 있는 경우)
                if 'content' in row and pd.notna(row['content']) and row['content'] != '':
                    content_chunks = chunk_text(row['content'], chunk_size, overlap)
                    if content_chunks:  # 청크가 있는 경우에만 생성
                        session.execute_write(create_content_nodes, article_id, content_chunks, article_data)

                # 3. Media 노드와 관계 생성
                if 'source' in row:
                    session.execute_write(create_media_node_and_relationship, article_id, row['source'])
                # 4. Category 노드와 관계 생성
                if 'category' in row:
                    session.execute_write(create_category_node_and_relationship, article_id, row['category'])

                # 진행상황 출력
                if (idx + 1) % 10 == 0:
                    print(f"진행률: {idx + 1}/{len(df)} ({((idx + 1)/len(df)*100):.1f}%)")

            except Exception as e:
                print(f"기사 {idx} 처리 중 오류 발생: {e}")
                continue

def main():
    # 1. 엑셀 파일 찾기
    list_of_files = glob.glob('Articles_*.xlsx') 
    if not list_of_files:
        print("처리할 Articles 엑셀 파일이 없습니다.")
        return
    
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"최신 데이터 파일 로드 중: {latest_file}")
    
    try:
        df = pd.read_excel(latest_file)
    except Exception as e:
        print(f"엑셀 파일 읽기 실패: {e}")
        return

    # 2. Neo4j 연결
    try:
        driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)
        driver.verify_connectivity()
        print("Neo4j 데이터베이스 연결 성공")
    except Exception as e:
        print(f"Neo4j 연결 실패: {e}")
        return

    try:
        # 3. DB 초기화 및 제약조건 생성
        print("데이터베이스 초기화 중...")
        with driver.session() as session:
            session.execute_write(clear_database)
            session.execute_write(create_constraints)
        
        # 4. 그래프 구축
        print("그래프 구축 시작...")
        build_graph_from_dataframe(driver, df)
        print("그래프 구축 완료!")

    finally:
        driver.close()

if __name__ == "__main__":
    main()
