import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { PGVectorStore } from "langchain/vectorstores/pg";

const pgConfig = {
  database: "db",
  user: "postgres",
  password: "postgres",
  max: 20,
  host: "localhost",
  port: 5432,
};

const embedding = new OpenAIEmbeddings();
const vectorStore = new PGVectorStore(embedding, {
  /**
   * You can pass either a pg-promise client, postgres url or postgres configuration object.
   */
  postgresConnectionOptions: pgConfig,
  useHnswIndex: false,
  tableName: "pg_embeddings",
  columns: {
    contentColumnName: "content", // defaults to content
  },
  /**
   * These are extra columns that will be added to the table or should
   * already by on the table. Useful for filtering + joining.
   */
  extraColumns: [
    {
      name: "extra_stuff", // column name
      type: "text", // column type based on postgresql types
      returned: true, // should it be returned when doing similarity search?
      notNull: true, // should it be not null?
    },
    {
      name: "some_table_id",
      type: "integer",
      returned: false,
      /**
       * If you want to reference another column in another table with a
       * foreign key, you can do so by setting references to the table
       * you want to refer. It can also be an object like so:
       * {
       *  table: "some_table",
       *  column: "id"
       * }
       */
      references: "some_table",
    },
  ],
  /**
   * You can either use pgvector or pg_embedding pg extention. You can customize the
   * metric, and the dimensionality of the vector by creating PGEmbeddingExt or PGVectorExt
   * like so:
   * import { PGEmbeddingExt, PGVectorExt } from "langchain/vectorstores/pg"
   * new PGEmbeddingExt({
   *   dims: 1536, // defaults to 1536
   *   metric: "cosine", // can be "cosine", "l2" and "manhattan" for pgembedding, and "cosine", "l2" and "inner_product" for pgvector
   *   pgInstance: pg, // should be an instance of pg-promise
   * })
   *
   * PGEmbeddingExt or PGVectorExt are implementations of abstract class
   * PostgresEmbeddingExtension which can be extended. See the source code
   * for implementation details.
   *
   * see /examples/src/indexes/vector_stores/pg_vectorstore/pgvector/Dockerfile
   * and /examples/src/indexes/vector_stores/pg_vectorstore/pgembedding/Dockerfile
   * if you need pointers on how to add pgvector or pgembedding to your postgresql.
   */
  pgExtensionOpts: {
    // optional, defaults to pgvector with 1536 dims and cosine metric
    type: "pgvector", // can be "pgvector", "pgembedding"
    dims: 1536, // optional, defaults to 1536
    metric: "cosine", // optional, defaults to "cosine". can be "cosine", "l2" and "manhattan" for pgembedding, and "cosine", "l2" and "inner_product" for pgvector
  }, // it can also be PGEmbeddingExt or PGVectorExt instance instead of an object
});

/**
 * ..then add tables to the database.
 */
await vectorStore.ensureTableInDatabase();

/**
 * ...and then create HNSW index
 */
await vectorStore.buildIndex("hnsw_index", {
  m: 16, // the max number of connections per layer (16 by default)
  efConstruction: 200, // the size of the dynamic candidate list for constructing the graph (64 by default)
  efSearch: 200, // Influences the trade-off between query accuracy (recall) and speed. A higher efsearch value increases accuracy at the cost of speed. This value should be equal to or larger than k, which is the number of nearest neighbors you want your search to return (defined by the LIMIT clause in your SELECT query).
});

const createdAt = new Date().toISOString();
const docs = [
  {
    pageContent:
      "This is a long text, but it actually means something because vector database does not understand Lorem Ipsum. So I would need to expand upon the notion of quantum fluff, a theorectical concept where subatomic particles coalesce to form transient multidimensional spaces. Yet, this abstraction holds no real-world application or comprehensible meaning, reflecting a cosmic puzzle.",
    metadata: { b: 1, c: 10, stuff: "right", created_at: createdAt },
  },
  {
    pageContent:
      "This is a long text, but it actually means something because vector database does not understand Lorem Ipsum. So I would need to proceed by discussing the echo of virtual tweets in the binary corridors of the digital universe. Each tweet, like a pixelated canary, hums in an unseen frequency, a fascinatingly perplexing phenomenon that, while conjuring vivid imagery, lacks any concrete implication or real-world relevance, portraying a paradox of multidimensional spaces in the age of cyber folklore.",
    metadata: { b: 2, c: 9, stuff: "right", created_at: createdAt },
  },
];

/**
 * If you define extraColumns, you can pass them under extraColumns.
 * If notNull is true, then you must pass the extra column.
 */
await vectorStore.addDocuments(docs, {
  extraColumns: [{ extra_stuff: 1 }, { extra_stuff: 2 }],
});

/**
 * Similarity search can either be based om column filtering or
 * metadata filtering (exclusively). The shape of the filter query
 * is exactly the same as ChromaDB and Pinecone filter query.
 * see https://docs.trychroma.com/usage-guide#using-where-filters
 */
await vectorStore.similaritySearch("This is a long text", 1, {
  columnFilter: {
    extra_stuff: "hello 1",
  },
});
await vectorStore.similaritySearch("This is a long text", 1, {
  metadataFilter: {
    stuff: "right",
  },
});

/**
 * If the filter does not contain "columnFilter" or "metadataFilter",
 * then it will be treated as a metadata filter. Specifically, it will
 * be treated as a metadata filter with "contains" operator.
 * e.g. WHERE metadata @> '{"b": 2, "c": 9"}'
 */
await vectorStore.similaritySearch("This is a long text", 1, {
  b: 2,
  c: 9,
});

/**
 * You can also do a postgres full-text search + similarity search.
 */
await vectorStore.similaritySearch("This is a long text", 1, {
  columnFilter: {
    /**
     * content is the column name for pageContent in the DB, but can be changed to anything
     * by setting pageContentColumn when defining it in KnexVectorStoreArgs.
     */
    content: {
      $textSearch: { query: "'vector' & 'database'", config: "english" },
    },
  },
});

/**
 * You can also do join and then filter on the joined columns.
 */
await vectorStore.similaritySearch("This is a long text", 1, {
  columnFilter: {
    "some_table.value": "hello 1",
  },
  join: {
    op: "JOIN",
    table: "some_table",
    on: [
      {
        left: "some_table.embeddings_id",
        operator: "=",
        right: "pg_embeddings.id",
      },
    ],
  },
});

/**
 * ...or do multiple joins.
 */
await vectorStore.similaritySearch("This is a long text", 1, {
  columnFilter: {
    "some_other_table.some_value": "hello 1",
  },
  join: [
    {
      op: "JOIN",
      table: "some_table",
      on: [
        {
          left: "some_table.embeddings_id",
          operator: "=",
          right: "pg_embeddings.id",
        },
      ],
    },
    {
      op: "INNER JOIN",
      table: "some_other_table",
      on: [
        {
          left: "some_other_table.id",
          operator: "=",
          right: "some_table.other_id",
        },
      ],
    },
  ],
});

/**
 * And everything you can do with similaritySearch, you can
 * also do with maxMarginalRelevanceSearch.
 */
await vectorStore.maxMarginalRelevanceSearch("This is a long text", {
  k: 3,
  fetchK: 7,
  filter: {
    columnFilter: {
      "some_other_table.some_value": "hello 1",
    },
    join: [
      {
        op: "JOIN",
        table: "some_table",
        on: [
          {
            left: "some_table.embeddings_id",
            operator: "=",
            right: "pg_embeddings.id",
          },
        ],
      },
      {
        op: "INNER JOIN",
        table: "some_other_table",
        on: [
          {
            left: "some_other_table.id",
            operator: "=",
            right: "some_table.other_id",
          },
        ],
      },
    ],
  },
});

/**
 * You can also drop the index.
 */
await vectorStore.dropIndex("hnsw_index");
