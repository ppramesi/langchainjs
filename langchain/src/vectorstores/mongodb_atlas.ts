import type {
  Collection,
  MongoClientOptions,
  Document as MongoDBDocument,
} from "mongodb";
import {
  BaseVectorStoreFields,
  VectorStore,
  VectorStoreInput,
} from "./base.js";
import { Embeddings } from "../embeddings/base.js";
import { Document } from "../document.js";

export type MongoDBAtlasVectorSearchLibArgs = (
  | {
      collection: Collection<MongoDBDocument>;
    }
  | {
      uri: string;
      dbName: string;
      collectionName: string;
      mongoOptions?: MongoClientOptions;
    }
) & {
  indexName?: string;
  textKey?: string;
  embeddingKey?: string;
};

export class MongoDBAtlasVectorSearch extends VectorStore {
  get lc_secrets(): { [key: string]: string } | undefined {
    return {
      uri: "MONGODB_ATLAS_URI",
    };
  }

  declare FilterType: MongoDBDocument;

  collection?: Collection<MongoDBDocument>;

  uri?: string;

  mongoOptions?: MongoClientOptions;

  dbName?: string;

  collectionName?: string;

  indexName: string;

  textKey: string;

  embeddingKey: string;

  vectorstoreType(): string {
    return "mongodb_atlas";
  }

  constructor(fields: VectorStoreInput<MongoDBAtlasVectorSearchLibArgs>);

  constructor(embeddings: Embeddings, args: MongoDBAtlasVectorSearchLibArgs);

  constructor(
    fieldsOrEmbeddings: BaseVectorStoreFields<MongoDBAtlasVectorSearchLibArgs>,
    extrArgs?: MongoDBAtlasVectorSearchLibArgs
  ) {
    const {
      embeddings,
      args: { ...args },
    } = MongoDBAtlasVectorSearch.unrollFields<MongoDBAtlasVectorSearchLibArgs>(
      fieldsOrEmbeddings,
      extrArgs
    );
    super({ embeddings, ...args });
    if ("collection" in args) {
      this.collection = args.collection;
      this.lc_serializable = false;
    } else if ("uri" in args && "dbName" in args && "collectionName" in args) {
      this.uri = args.uri;
      this.mongoOptions = args.mongoOptions;
      this.dbName = args.dbName;
      this.collectionName = args.collectionName;
      this.lc_serializable = true;
    } else {
      throw new Error(
        "MongoDBAtlasVectorSearch requires either MongoDB Collection instance or uri, db name and collection name."
      );
    }
    this.indexName = args.indexName || "default";
    this.textKey = args.textKey || "text";
    this.embeddingKey = args.embeddingKey || "embedding";
  }

  async ensureCollection() {
    if (!this.collection) {
      if (this.uri && this.collectionName && this.dbName) {
        const { MongoClient } =
          await MongoDBAtlasVectorSearch.importCreateCollection();
        const client = new MongoClient(this.uri, this.mongoOptions);
        return client.db(this.dbName).collection(this.collectionName);
      } else {
        throw new Error("Cannot find url or private key");
      }
    }
    return this.collection;
  }

  async addVectors(vectors: number[][], documents: Document[]): Promise<void> {
    const docs = vectors.map((embedding, idx) => ({
      [this.textKey]: documents[idx].pageContent,
      [this.embeddingKey]: embedding,
      ...documents[idx].metadata,
    }));
    await (await this.ensureCollection()).insertMany(docs);
  }

  async addDocuments(documents: Document[]): Promise<void> {
    const texts = documents.map(({ pageContent }) => pageContent);
    return this.addVectors(
      await this.embeddings.embedDocuments(texts),
      documents
    );
  }

  async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    preFilter?: MongoDBDocument,
    postFilterPipeline?: MongoDBDocument[]
  ): Promise<[Document, number][]> {
    const knnBeta: MongoDBDocument = {
      vector: query,
      path: this.embeddingKey,
      k,
    };
    if (preFilter) {
      knnBeta.filter = preFilter;
    }
    const pipeline: MongoDBDocument[] = [
      {
        $search: {
          index: this.indexName,
          knnBeta,
        },
      },
      {
        $project: {
          [this.embeddingKey]: 0,
          score: { $meta: "searchScore" },
        },
      },
    ];
    if (postFilterPipeline) {
      pipeline.push(...postFilterPipeline);
    }
    const results = (await this.ensureCollection()).aggregate(pipeline);

    const ret: [Document, number][] = [];
    for await (const result of results) {
      const text = result[this.textKey];
      delete result[this.textKey];
      const { score, ...metadata } = result;
      ret.push([new Document({ pageContent: text, metadata }), score]);
    }

    return ret;
  }

  async similaritySearch(
    query: string,
    k: number,
    preFilter?: MongoDBDocument,
    postFilterPipeline?: MongoDBDocument[]
  ): Promise<Document[]> {
    const results = await this.similaritySearchVectorWithScore(
      await this.embeddings.embedQuery(query),
      k,
      preFilter,
      postFilterPipeline
    );
    return results.map((result) => result[0]);
  }

  static async fromTexts(
    texts: string[],
    metadatas: object[] | object,
    embeddings: Embeddings,
    dbConfig: MongoDBAtlasVectorSearchLibArgs
  ): Promise<MongoDBAtlasVectorSearch> {
    const docs: Document[] = [];
    for (let i = 0; i < texts.length; i += 1) {
      const metadata = Array.isArray(metadatas) ? metadatas[i] : metadatas;
      const newDoc = new Document({
        pageContent: texts[i],
        metadata,
      });
      docs.push(newDoc);
    }
    return MongoDBAtlasVectorSearch.fromDocuments(docs, embeddings, dbConfig);
  }

  static async fromDocuments(
    docs: Document[],
    embeddings: Embeddings,
    dbConfig: MongoDBAtlasVectorSearchLibArgs
  ): Promise<MongoDBAtlasVectorSearch> {
    const instance = new this(embeddings, dbConfig);
    await instance.addDocuments(docs);
    return instance;
  }

  static async importCreateCollection() {
    try {
      const { MongoClient } = await import("mongodb");
      return { MongoClient };
    } catch (error) {
      throw new Error(
        "Please install Supabase as a dependency with, e.g. `npm install -S Supabase`"
      );
    }
  }
}
