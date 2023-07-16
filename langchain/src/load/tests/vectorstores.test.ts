import { test } from "@jest/globals";
import { ChromaClient } from "chromadb";
import { Chroma } from "../../vectorstores/chroma.js";
// import { ElasticVectorSearch } from "../../vectorstores/elasticsearch.js";
// import { FaissStore } from "../../vectorstores/faiss.js";
// import { HNSWLib } from "../../vectorstores/hnswlib.js";
// import { LanceDB } from "../../vectorstores/lancedb.js";
// import { MemoryVectorStore } from "../../vectorstores/memory.js";
// import { Milvus } from "../../vectorstores/milvus.js";
// import { MongoDBAtlasVectorSearch } from "../../vectorstores/mongodb_atlas.js";
// import { MongoVectorStore } from "../../vectorstores/mongo.js";
// import { MyScaleStore } from "../../vectorstores/myscale.js";
// import { OpenSearchVectorStore } from "../../vectorstores/opensearch.js";
// import { PineconeStore } from "../../vectorstores/pinecone.js";
// import { PrismaVectorStore } from "../../vectorstores/prisma.js";
// import { QdrantVectorStore } from "../../vectorstores/qdrant.js";
// import { RedisVectorStore } from "../../vectorstores/redis.js";
// import { SingleStoreVectorStore } from "../../vectorstores/singlestore.js";
// import { SupabaseVectorStore } from "../../vectorstores/supabase.js";
// import { TigrisVectorStore } from "../../vectorstores/tigris.js";
// import { TypeORMVectorStore } from "../../vectorstores/typeorm.js";
// import { Typesense } from "../../vectorstores/typesense.js";
// import { VectaraStore } from "../../vectorstores/vectara.js";
// import { WeaviateStore } from "../../vectorstores/weaviate.js";
import { FakeEmbeddings } from "../../embeddings/fake.js";

test("Chroma serde, serializable", () => {
  const chromaStore = new Chroma({
    embeddings: new FakeEmbeddings(),
    url: "http://localhost:8000",
    collectionName: "test-collection",
  });
  console.log(JSON.stringify(chromaStore, null, 2));
});

test("Chroma serde, unserializable", () => {
  const chromaStore = new Chroma({
    embeddings: new FakeEmbeddings(),
    index: new ChromaClient({
      path: "http://localhost:8000",
    }),
  });
  console.log(JSON.stringify(chromaStore, null, 2));
});
