import { test } from "@jest/globals";
import { ChromaClient } from "chromadb";
import { Client } from "@elastic/elasticsearch";
import { Chroma } from "../../vectorstores/chroma.js";
import { ElasticVectorSearch } from "../../vectorstores/elasticsearch.js";
import { FaissStore } from "../../vectorstores/faiss.js";
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
import { OpenAIEmbeddings } from "../../embeddings/openai.js";
import { load } from "../index.js";

test("Chroma serde, serializable", async () => {
  const embeddings = new OpenAIEmbeddings();
  const store = new Chroma({
    embeddings,
    url: "http://localhost:8000",
    collectionName: "test-collection",
  });
  const str = JSON.stringify(store, null, 2);
  const vs = await load<Chroma>(
    str,
    { COHERE_API_KEY: "cohere-key" },
    { "langchain/vectorstores/chroma": { Chroma } }
  );
  expect(vs).toBeInstanceOf(Chroma);
  expect(vs.embeddings).toBeInstanceOf(OpenAIEmbeddings);
  const str2 = JSON.stringify(vs, null, 2);
  expect(str2).toEqual(str);
  const embStr = JSON.stringify(embeddings);
  const embStr2 = JSON.stringify(vs.embeddings);
  expect(embStr).toEqual(embStr2);
});

test("Chroma serde, unserializable", () => {
  const embeddings = new OpenAIEmbeddings();
  const store = new Chroma({
    embeddings,
    index: new ChromaClient({
      path: "http://localhost:8000",
    }),
  });
  const str = JSON.stringify(store, null, 2);
  const parsed = JSON.parse(str);
  expect(parsed.type).toEqual("not_implemented");
});

test("Elasticsearch serde, unserializable", () => {
  /* eslint-disable @typescript-eslint/no-explicit-any */
  const config: any = {
    node: "http://fakeurl.com",
  };
  config.auth = {
    username: "process.env.ELASTIC_USERNAME",
    password: "process.env.ELASTIC_PASSWORD",
  };
  const client = new Client(config);

  const indexName = "test_index";
  const embeddings = new OpenAIEmbeddings();
  const store = new ElasticVectorSearch({ embeddings, client, indexName });
  const str = JSON.stringify(store, null, 2);
  const parsed = JSON.parse(str);
  expect(parsed.type).toEqual("not_implemented");
});

test("Faiss serde, unserializable", async () => {
  const store = await FaissStore.fromTexts(
    ["Hello world", "Bye bye", "hello nice world"],
    [{ id: 2 }, { id: 1 }, { id: 3 }],
    new OpenAIEmbeddings()
  );
  const str = JSON.stringify(store, null, 2);
  const parsed = JSON.parse(str);
  expect(parsed.type).toEqual("not_implemented");
});
