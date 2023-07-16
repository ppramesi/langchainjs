/* eslint-disable @typescript-eslint/no-explicit-any */
import { Embeddings } from "../embeddings/base.js";
import { Document } from "../document.js";
import { BaseRetriever, BaseRetrieverInput } from "../schema/retriever.js";
import { Serializable } from "../load/serializable.js";
import {
  CallbackManagerForRetrieverRun,
  Callbacks,
} from "../callbacks/manager.js";

type AddDocumentOptions = Record<string, any>;

export interface VectorStoreRetrieverInput<V extends VectorStore>
  extends BaseRetrieverInput {
  vectorStore: V;
  k?: number;
  filter?: V["FilterType"];
}

export class VectorStoreRetriever<
  V extends VectorStore = VectorStore
> extends BaseRetriever {
  lc_serializable = true;

  get lc_namespace() {
    return ["langchain", "retrievers", this._vectorstoreType()];
  }

  vectorStore: V;

  k = 4;

  filter?: V["FilterType"];

  _vectorstoreType(): string {
    return this.vectorStore.vectorstoreType();
  }

  constructor(fields: VectorStoreRetrieverInput<V>) {
    super(fields);
    this.vectorStore = fields.vectorStore;
    this.k = fields.k ?? this.k;
    this.filter = fields.filter;
  }

  async _getRelevantDocuments(
    query: string,
    runManager?: CallbackManagerForRetrieverRun
  ): Promise<Document[]> {
    return this.vectorStore.similaritySearch(
      query,
      this.k,
      this.filter,
      runManager?.getChild("vectorstore")
    );
  }

  async addDocuments(
    documents: Document[],
    options?: AddDocumentOptions
  ): Promise<string[] | void> {
    return this.vectorStore.addDocuments(documents, options);
  }
}

export type VectorStoreInput<T extends { [k: string]: any }> = T & {
  embeddings: Embeddings;
};

export type BaseVectorStoreFields<T extends { [k: string]: any }> =
  | VectorStoreInput<T>
  | Embeddings;

export abstract class VectorStore extends Serializable {
  declare FilterType: object;

  lc_namespace = ["langchain", "vector_stores", this.vectorstoreType()];

  embeddings: Embeddings;

  constructor(fields: VectorStoreInput<{ [k: string]: any }>);

  constructor(embeddings: Embeddings, dbConfig: { [k: string]: any });

  constructor(
    fieldsOrEmbeddings: BaseVectorStoreFields<{ [k: string]: any }>,
    extrArgs?: { [k: string]: any }
  ) {
    const { embeddings, args } = VectorStore.unrollFields<{ [k: string]: any }>(
      fieldsOrEmbeddings,
      extrArgs
    );
    super({ embeddings, ...args });
    this.embeddings = embeddings;
  }

  abstract vectorstoreType(): string;

  abstract addVectors(
    vectors: number[][],
    documents: Document[],
    options?: AddDocumentOptions
  ): Promise<string[] | void>;

  abstract addDocuments(
    documents: Document[],
    options?: AddDocumentOptions
  ): Promise<string[] | void>;

  async delete(_params?: Record<string, any>): Promise<void> {
    throw new Error("Not implemented.");
  }

  abstract similaritySearchVectorWithScore(
    query: number[],
    k: number,
    filter?: this["FilterType"]
  ): Promise<[Document, number][]>;

  async similaritySearch(
    query: string,
    k = 4,
    filter: this["FilterType"] | undefined = undefined,
    _callbacks: Callbacks | undefined = undefined // implement passing to embedQuery later
  ): Promise<Document[]> {
    const results = await this.similaritySearchVectorWithScore(
      await this.embeddings.embedQuery(query),
      k,
      filter
    );

    return results.map((result) => result[0]);
  }

  async similaritySearchWithScore(
    query: string,
    k = 4,
    filter: this["FilterType"] | undefined = undefined,
    _callbacks: Callbacks | undefined = undefined // implement passing to embedQuery later
  ): Promise<[Document, number][]> {
    return this.similaritySearchVectorWithScore(
      await this.embeddings.embedQuery(query),
      k,
      filter
    );
  }

  static fromTexts(
    _texts: string[],
    _metadatas: object[] | object,
    _embeddings: Embeddings,
    _dbConfig: Record<string, any>
  ): Promise<VectorStore> {
    throw new Error(
      "the Langchain vectorstore implementation you are using forgot to override this, please report a bug"
    );
  }

  static fromDocuments(
    _docs: Document[],
    _embeddings: Embeddings,
    _dbConfig: Record<string, any>
  ): Promise<VectorStore> {
    throw new Error(
      "the Langchain vectorstore implementation you are using forgot to override this, please report a bug"
    );
  }

  static unrollFields<T extends { [k: string]: any }>(
    fieldsOrEmbeddings: BaseVectorStoreFields<T>,
    extrArgs?: T
  ) {
    let embeddings: Embeddings;
    let args: T;
    if (
      "embeddings" in fieldsOrEmbeddings &&
      // eslint-disable-next-line no-instanceof/no-instanceof
      fieldsOrEmbeddings.embeddings instanceof Embeddings
    ) {
      const { embeddings: tempEmbeddings, ...tempArgs } = fieldsOrEmbeddings;
      return { embeddings: tempEmbeddings, args: tempArgs };
      // eslint-disable-next-line no-instanceof/no-instanceof
    } else if (fieldsOrEmbeddings instanceof Embeddings && extrArgs) {
      args = extrArgs;
      embeddings = fieldsOrEmbeddings;
      return { embeddings, args };
    } else {
      throw new Error(
        "Second argument is required if first parameter is an embedding"
      );
    }
  }

  asRetriever(
    k?: number,
    filter?: this["FilterType"]
  ): VectorStoreRetriever<this> {
    return new VectorStoreRetriever({ vectorStore: this, k, filter });
  }
}

export abstract class SaveableVectorStore extends VectorStore {
  abstract save(directory: string): Promise<void>;

  static load(
    _directory: string,
    _embeddings: Embeddings
  ): Promise<SaveableVectorStore> {
    throw new Error("Not implemented");
  }
}
