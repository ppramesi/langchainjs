import * as uuid from "uuid";
import flatten from "flat";

import {
  BaseVectorStoreFields,
  VectorStore,
  VectorStoreInput,
} from "./base.js";
import { Embeddings } from "../embeddings/base.js";
import { Document } from "../document.js";

// eslint-disable-next-line @typescript-eslint/ban-types, @typescript-eslint/no-explicit-any
type PineconeMetadata = Record<string, any>;

type VectorOperationsApi = ReturnType<
  import("@pinecone-database/pinecone").PineconeClient["Index"]
>;

export type PineconeLibArgs = (
  | {
      pineconeIndex: VectorOperationsApi;
    }
  | {
      environment: string;
      apiKey: string;
      indexName?: string;
    }
) & {
  textKey?: string;
  namespace?: string;
  filter?: PineconeMetadata;
};

export type PineconeDeleteParams = {
  ids?: string[];
  deleteAll?: boolean;
  namespace?: string;
};

export class PineconeStore extends VectorStore {
  get lc_secrets(): { [key: string]: string } | undefined {
    return {
      apiKey: "PINECONE_API_KEY",
    };
  }

  declare FilterType: PineconeMetadata;

  textKey: string;

  namespace?: string;

  pineconeIndex?: VectorOperationsApi;

  environment?: string;

  apiKey?: string;

  filter?: PineconeMetadata;

  indexName?: string;

  vectorstoreType(): string {
    return "pinecone";
  }

  constructor(fields: VectorStoreInput<PineconeLibArgs>);

  constructor(embeddings: Embeddings, args: PineconeLibArgs);

  constructor(
    fieldsOrEmbeddings: BaseVectorStoreFields<PineconeLibArgs>,
    extrArgs?: PineconeLibArgs
  ) {
    const {
      embeddings,
      args: { ...args },
    } = PineconeStore.unrollFields<PineconeLibArgs>(
      fieldsOrEmbeddings,
      extrArgs
    );
    super({ embeddings, ...args });
    this.embeddings = embeddings;
    this.namespace = args.namespace;
    if ("pineconeIndex" in args) {
      this.pineconeIndex = args.pineconeIndex;
      this.lc_serializable = false;
    } else if (
      "apiKey" in args &&
      "environment" in args &&
      "indexName" in args
    ) {
      this.apiKey = args.apiKey;
      this.environment = args.environment;
      this.indexName = args.indexName;
      this.lc_serializable = true;
    } else {
      throw new Error(
        "MongoDBAtlasVectorSearch requires either MongoDB Collection instance or uri, db name and collection name."
      );
    }
    this.textKey = args.textKey ?? "text";
    this.filter = args.filter;
  }

  async ensureIndex() {
    if (!this.pineconeIndex) {
      if (this.apiKey && this.environment && this.indexName) {
        const { PineconeClient } = await PineconeStore.importPineconeClient();
        const client = new PineconeClient();
        return client.Index(this.indexName);
      } else {
        throw new Error("Cannot find api key or environment");
      }
    }
    return this.pineconeIndex;
  }

  async addDocuments(
    documents: Document[],
    options?: { ids?: string[] } | string[]
  ) {
    const texts = documents.map(({ pageContent }) => pageContent);
    return this.addVectors(
      await this.embeddings.embedDocuments(texts),
      documents,
      options
    );
  }

  async addVectors(
    vectors: number[][],
    documents: Document[],
    options?: { ids?: string[] } | string[]
  ) {
    const ids = Array.isArray(options) ? options : options?.ids;
    const documentIds = ids == null ? documents.map(() => uuid.v4()) : ids;
    const pineconeVectors = vectors.map((values, idx) => {
      // Pinecone doesn't support nested objects, so we flatten them
      const documentMetadata = { ...documents[idx].metadata };
      // preserve string arrays which are allowed
      const stringArrays: Record<string, string[]> = {};
      for (const key of Object.keys(documentMetadata)) {
        if (
          Array.isArray(documentMetadata[key]) &&
          // eslint-disable-next-line @typescript-eslint/ban-types, @typescript-eslint/no-explicit-any
          documentMetadata[key].every((el: any) => typeof el === "string")
        ) {
          stringArrays[key] = documentMetadata[key];
          delete documentMetadata[key];
        }
      }
      const metadata: {
        [key: string]: string | number | boolean | string[] | null;
      } = {
        ...flatten(documentMetadata),
        ...stringArrays,
        [this.textKey]: documents[idx].pageContent,
      };
      // Pinecone doesn't support null values, so we remove them
      for (const key of Object.keys(metadata)) {
        if (metadata[key] == null) {
          delete metadata[key];
        } else if (
          typeof metadata[key] === "object" &&
          Object.keys(metadata[key] as unknown as object).length === 0
        ) {
          delete metadata[key];
        }
      }

      return {
        id: documentIds[idx],
        metadata,
        values,
      };
    });

    // Pinecone recommends a limit of 100 vectors per upsert request
    const chunkSize = 50;
    for (let i = 0; i < pineconeVectors.length; i += chunkSize) {
      const chunk = pineconeVectors.slice(i, i + chunkSize);
      await (
        await this.ensureIndex()
      ).upsert({
        upsertRequest: {
          vectors: chunk,
          namespace: this.namespace,
        },
      });
    }
    return documentIds;
  }

  async delete(params: PineconeDeleteParams): Promise<void> {
    const { namespace = this.namespace, deleteAll, ids, ...rest } = params;
    if (deleteAll) {
      await (
        await this.ensureIndex()
      ).delete1({
        deleteAll: true,
        namespace,
        ...rest,
      });
    } else if (ids) {
      const batchSize = 1000;
      for (let i = 0; i < ids.length; i += batchSize) {
        const batchIds = ids.slice(i, i + batchSize);
        await (
          await this.ensureIndex()
        ).delete1({
          ids: batchIds,
          namespace,
          ...rest,
        });
      }
    } else {
      throw new Error("Either ids or delete_all must be provided.");
    }
  }

  async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    filter?: PineconeMetadata
  ): Promise<[Document, number][]> {
    if (filter && this.filter) {
      throw new Error("cannot provide both `filter` and `this.filter`");
    }
    const _filter = filter ?? this.filter;
    const results = await (
      await this.ensureIndex()
    ).query({
      queryRequest: {
        includeMetadata: true,
        namespace: this.namespace,
        topK: k,
        vector: query,
        filter: _filter,
      },
    });

    const result: [Document, number][] = [];

    if (results.matches) {
      for (const res of results.matches) {
        const { [this.textKey]: pageContent, ...metadata } = (res.metadata ??
          {}) as PineconeMetadata;
        if (res.score) {
          result.push([new Document({ metadata, pageContent }), res.score]);
        }
      }
    }

    return result;
  }

  static async fromTexts(
    texts: string[],
    metadatas: object[] | object,
    embeddings: Embeddings,
    dbConfig:
      | {
          /**
           * @deprecated Use pineconeIndex instead
           */
          pineconeClient: VectorOperationsApi;
          textKey?: string;
          namespace?: string | undefined;
        }
      | PineconeLibArgs
  ): Promise<PineconeStore> {
    const docs: Document[] = [];
    for (let i = 0; i < texts.length; i += 1) {
      const metadata = Array.isArray(metadatas) ? metadatas[i] : metadatas;
      const newDoc = new Document({
        pageContent: texts[i],
        metadata,
      });
      docs.push(newDoc);
    }
    let args: PineconeLibArgs;
    if ("pineconeIndex" in dbConfig || "pineconeClient" in dbConfig) {
      args = {
        pineconeIndex:
          "pineconeIndex" in dbConfig
            ? dbConfig.pineconeIndex
            : dbConfig.pineconeClient,
        textKey: dbConfig.textKey,
        namespace: dbConfig.namespace,
      };
    } else {
      args = dbConfig;
    }
    return PineconeStore.fromDocuments(docs, embeddings, args);
  }

  static async fromDocuments(
    docs: Document[],
    embeddings: Embeddings,
    dbConfig: PineconeLibArgs
  ): Promise<PineconeStore> {
    const args = dbConfig;
    args.textKey = dbConfig.textKey ?? "text";

    const instance = new this(embeddings, args);
    await instance.addDocuments(docs);
    return instance;
  }

  static async fromExistingIndex(
    embeddings: Embeddings,
    dbConfig: PineconeLibArgs
  ): Promise<PineconeStore> {
    const instance = new this(embeddings, dbConfig);
    return instance;
  }

  static async importPineconeClient() {
    try {
      const { PineconeClient } = await import("@pinecone-database/pinecone");
      return { PineconeClient };
    } catch (error) {
      throw new Error(
        "Please install pinecone database as a dependency with, e.g. `npm install -S @pinecone-database/pinecone`"
      );
    }
  }
}
