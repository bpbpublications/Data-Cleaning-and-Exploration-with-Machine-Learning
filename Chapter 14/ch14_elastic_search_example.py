from elasticsearch import Elasticsearch

# Connect to a distributed Elasticsearch cluster
es = Elasticsearch(
    ["http://node1:9200", "http://node2:9200", "http://node3:9200"],
    timeout=30
)

# Query the index with distributed search
query = {
    "query": {
        "match": {"product_description": "smartphone"}
    }
}
response = es.search(index="products", body=query)
print(response)
