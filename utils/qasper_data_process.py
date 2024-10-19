import json

input_file = '/home/zhangyusi/RAG_data/qasper/dummy_data/qasper-test-and-evaluator-v0.3/qasper-test-v0.3.json'
output_file = '/home/zhangyusi/RAG_data/qasper/dummy_data/qasper-test-and-evaluator-v0.3/qasper-test-v0.3_extract.json'

def process_article(article_id, article_data):
    
    title = article_data.get("title", "")
    abstract = article_data.get("abstract", "")
    
    full_text_sections = article_data.get("full_text", [])
    full_text = ""
    for section in full_text_sections:
        section_name = section.get("section_name", "")
        if section_name == None:
            section_name = '\n'
        paragraphs = section.get("paragraphs", [])
        full_text += section_name + "\n" + "\n".join(paragraphs) + "\n"
    article_text = f"{title}\n{abstract}\n{full_text}"
    
    qas = article_data.get("qas", [])
    query_list = [{"question": qa.get("question", ""), "question_id": qa.get("question_id", "")} for qa in qas]
    
    return {
        "article_id": article_id,
        "article": article_text,
        "query_list": query_list
    }

def process_all_articles(data):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for article_id, article_data in data.items():
            processed_article = process_article(article_id, article_data)
            outfile.write(json.dumps(processed_article, ensure_ascii=False) + "\n")
            outfile.flush()

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

processed_articles = process_all_articles(data)
print(f"Process finished，data has been saved into {output_file}")
