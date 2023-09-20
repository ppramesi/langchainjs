import Knex from "knex";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { KnexVectorStore } from "langchain/vectorstores/knex";

const knex = Knex.knex({
  client: "postgresql",
  connection: {
    database: "db",
    user: "postgres",
    password: "postgres",
  },
  pool: {
    min: 0,
    max: 10,
  },
});

export async function run() {
  const embedding = new OpenAIEmbeddings();
  const vectorStore = new KnexVectorStore(embedding, {
    knex: knex,
    useHnswIndex: false,
    tableName: "knex_embeddings",
    pageContentColumn: "content", // defaults to "content"
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
         * If you want to reference a foreign key, you can do so by setting
         * references to the table you want to reference. It can also be an
         * object like so:
         * {
         *  table: "some_table",
         *  column: "id"
         * }
         */
        references: "some_table",
      },
    ],
    /**
     * You can either use pgvector or pg_embedding. You can customize the metric,
     * and the dimensionality of the vector by creating PGEmbeddingExt or PGVectorExt
     * like so:
     * import { PGEmbeddingExt, PGVectorExt } from "langchain/vectorstores/knex"
     * new PGEmbeddingExt({
     *   dims: 1536, // defaults to 1536
     *   metric: "cosine", // can be "cosine", "l2" and "manhattan" for pgembedding, and "cosine", "l2" and "inner_product" for pgvector
     *   knex: knex,
     * })
     */
    pgExtension: "pgvector", // defaults to pgvector, can be "pgvector", "pgembedding", PGEmbeddingExt or PGVectorExt instance
  });

  /**
   * ...then create HNSW index
   */
  await vectorStore.buildIndex("hnsw_index", {
    m: 16,
    efConstruction: 200,
    efSearch: 200,
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
   * metadata filtering. The shape of the filter query is exactly
   * the same as ChromaDB and Pinecone filter query.
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
   * You can also do joins and then filter on the joined table.
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
          right: "knex_embeddings.id",
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
            right: "knex_embeddings.id",
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
   * After you're done, you can drop the index.
   */
  await vectorStore.dropIndex("hnsw_index");
}
