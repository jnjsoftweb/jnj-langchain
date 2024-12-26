from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    'C:/JnJ-soft/Projects/external/km-classics/txt/8.txt',
    encoding='utf-8'
)
data = loader.load()

print(data[:500])