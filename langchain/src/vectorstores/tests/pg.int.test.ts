/* eslint-disable no-process-env */
import {
  test,
  expect,
  beforeAll,
  beforeEach,
  afterAll,
  afterEach,
} from "@jest/globals";
import pgPromise, { IDatabase } from "pg-promise";
import { OpenAIEmbeddings } from "../../embeddings/openai.js";
import { PGVectorStore } from "../pg.js";

/**
 * We're using two different postgres instances for each extension. Should setup with docker,
 * see https://github.com/ppramesi/vector-pg-tests/blob/main/database_pgvector/Dockerfile
 * and https://github.com/ppramesi/vector-pg-tests/blob/main/database_pgembedding/Dockerfile
 * for dockerfile configs.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let pgvsPgvector: IDatabase<any>;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let pgvsPgembedding: IDatabase<any>;

if (
  !process.env.POSTGRES_HOST ||
  !process.env.POSTGRES_PGVECTOR_DB ||
  !process.env.POSTGRES_PGVECTOR_USER ||
  !process.env.POSTGRES_PGVECTOR_PASSWORD ||
  !process.env.POSTGRES_PGVECTOR_PORT
) {
  throw new Error("PGVECTOR environment variables not set");
}

if (
  !process.env.POSTGRES_HOST ||
  !process.env.POSTGRES_PGEMBEDDING_DB ||
  !process.env.POSTGRES_PGEMBEDDING_USER ||
  !process.env.POSTGRES_PGEMBEDDING_PASSWORD ||
  !process.env.POSTGRES_PGEMBEDDING_PORT
) {
  throw new Error("PGEMBEDDING environment variables not set");
}

beforeAll(async () => {
  const pgp = pgPromise();
  pgvsPgvector = pgp({
    host: process.env.POSTGRES_HOST,
    database: process.env.POSTGRES_PGVECTOR_DB,
    user: process.env.POSTGRES_PGVECTOR_USER,
    password: process.env.POSTGRES_PGVECTOR_PASSWORD,
    port: Number(process.env.POSTGRES_PGVECTOR_PORT),
    max: 20,
  });
  pgvsPgembedding = pgp({
    host: process.env.POSTGRES_HOST,
    database: process.env.POSTGRES_PGEMBEDDING_DB,
    user: process.env.POSTGRES_PGEMBEDDING_USER,
    password: process.env.POSTGRES_PGEMBEDDING_PASSWORD,
    port: Number(process.env.POSTGRES_PGEMBEDDING_PORT),
    max: 20,
  });
});

beforeEach(async () => {
  /**
   * 🚨🚨🚨 WARNING WARNING WARNING 🚨🚨🚨
   * We're dropping pg_embeddings table first to make sure the tests
   * are idempotent. This means that if you have a table called
   * pg_embeddings in your database, it will be dropped.
   */
  await Promise.all([
    pgvsPgvector.none("DROP TABLE IF EXISTS pg_embeddings"),
    pgvsPgembedding.none("DROP TABLE IF EXISTS pg_embeddings"),
  ]);

  const embedding = new OpenAIEmbeddings();

  const pgvKnexVS = new PGVectorStore(embedding, {
    postgresConnectionOptions: pgvsPgvector,
    useHnswIndex: true,
    tableName: "pg_embeddings",
    columns: {
      contentColumnName: "content",
    },
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtensionOpts: { type: "pgvector", dims: 1536 },
  });
  const pgeKnexVS = new PGVectorStore(embedding, {
    postgresConnectionOptions: pgvsPgembedding,
    useHnswIndex: true,
    tableName: "pg_embeddings",
    columns: {
      contentColumnName: "content",
    },
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtensionOpts: { type: "pgembedding", dims: 1536 },
  });

  await Promise.all([
    pgvKnexVS.ensureTableInDatabase(),
    pgeKnexVS.ensureTableInDatabase(),
  ]);
});

afterEach(async () => {
  await Promise.all([
    pgvsPgvector.none("DROP TABLE IF EXISTS pg_embeddings"),
    pgvsPgembedding.none("DROP TABLE IF EXISTS pg_embeddings"),
  ]);
});

afterAll(async () => {
  await Promise.all([pgvsPgvector.$pool.end(), pgvsPgembedding.$pool.end()]);
});

test("Build index pgvector", async () => {
  const embedding = new OpenAIEmbeddings();
  const pgVS = new PGVectorStore(embedding, {
    postgresConnectionOptions: pgvsPgvector,
    useHnswIndex: true,
    tableName: "pg_embeddings",
    columns: {
      contentColumnName: "content",
    },
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtensionOpts: { type: "pgvector", dims: 1536 },
  });

  try {
    await pgVS.buildIndex("test_hnsw_index");
    expect(true).toBe(true);
  } catch (error) {
    expect(error).toBe(null);
  } finally {
    await pgVS.dropIndex("test_hnsw_index");
  }
});

test("MMR and Similarity Search Test pgvector", async () => {
  const embedding = new OpenAIEmbeddings();
  const pgVS = new PGVectorStore(embedding, {
    postgresConnectionOptions: pgvsPgvector,
    useHnswIndex: true,
    tableName: "pg_embeddings",
    columns: {
      contentColumnName: "content",
    },
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtensionOpts: { type: "pgvector", dims: 1536 },
  });

  const createdAt = new Date().getTime();

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
    {
      pageContent: "hello",
      metadata: { b: 1, c: 9, stuff: "right", created_at: createdAt },
    },
    {
      pageContent: "hello",
      metadata: { b: 1, c: 9, stuff: "wrong", created_at: createdAt },
    },
    {
      pageContent: "hi",
      metadata: { b: 2, c: 8, stuff: "right", created_at: createdAt },
    },
    {
      pageContent: "bye",
      metadata: { b: 3, c: 7, stuff: "right", created_at: createdAt },
    },
    {
      pageContent: "what's this",
      metadata: { b: 4, c: 6, stuff: "right", created_at: createdAt },
    },
  ];

  await pgVS.addDocuments(docs, {
    extraColumns: [
      { extra_stuff: "hello 1" },
      { extra_stuff: "hello 2" },
      { extra_stuff: "hello 3" },
      { extra_stuff: "hello 4" },
      { extra_stuff: "hello 5" },
      { extra_stuff: "hello 6" },
      { extra_stuff: "hello 7" },
    ],
  });

  const ssResults = await pgVS.similaritySearch("hello", 7);

  expect(ssResults.length).toBe(7);

  const mmrResults = await pgVS.maxMarginalRelevanceSearch("hello", {
    k: 3,
    fetchK: 7,
  });

  expect(mmrResults.length).toBe(3);
});

test("Building WHERE query test pgvector", () => {
  const embedding = new OpenAIEmbeddings();
  const pgVS = new PGVectorStore(embedding, {
    postgresConnectionOptions: pgvsPgvector,
    useHnswIndex: true,
    tableName: "pg_embeddings",
    columns: {
      contentColumnName: "content",
    },
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtensionOpts: { type: "pgvector", dims: 1536 },
  });

  const queryMetadata = pgVS.buildSqlFilterStr(
    {
      $or: [
        { stuff: { $eq: "hello" } },
        { hello: "stuff" },
        {
          $and: [
            { hello: "stuff" },
            {
              content: {
                $textSearch: {
                  query: "hello",
                  config: "english",
                  type: "plain",
                },
              },
            },
          ],
        },
      ],
    },
    "metadata"
  );

  expect(JSON.stringify(queryMetadata)).toBe(
    '["WHERE",{"query":"((($1:raw)::text = $2 OR ($3:raw)::text = $4 OR (($5:raw)::text = $6 AND to_tsvector($7, $8:raw) @@ plainto_tsquery($9, $10))))","values":["metadata->>\'stuff\'","hello","metadata->>\'hello\'","stuff","metadata->>\'hello\'","stuff","english","content","english","hello"]}]'
  );

  const queryColumn = pgVS.buildSqlFilterStr(
    {
      $or: [
        { stuff: { $eq: "hello" } },
        { hello: "stuff" },
        {
          $and: [
            { hello: "stuff" },
            {
              content: {
                $textSearch: {
                  query: "hello",
                  config: "english",
                  type: "plain",
                },
              },
            },
          ],
        },
      ],
    },
    "column"
  );

  expect(JSON.stringify(queryColumn)).toBe(
    '["WHERE",{"query":"((($1:raw) = $2 OR ($3:raw) = $4 OR (($5:raw) = $6 AND to_tsvector($7, $8:raw) @@ plainto_tsquery($9, $10))))","values":["stuff","hello","hello","stuff","hello","stuff","english","content","english","hello"]}]'
  );
});

test("MMR and Similarity Search with filter Test pgvector", async () => {
  const embedding = new OpenAIEmbeddings();
  const pgVS = new PGVectorStore(embedding, {
    postgresConnectionOptions: pgvsPgvector,
    useHnswIndex: true,
    tableName: "pg_embeddings",
    columns: {
      contentColumnName: "content",
    },
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtensionOpts: { type: "pgvector", dims: 1536 },
  });

  const createdAt = new Date().getTime();

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
    {
      pageContent: "hello",
      metadata: { b: 1, c: 9, stuff: "right", created_at: createdAt },
    },
    {
      pageContent: "hello",
      metadata: { b: 1, c: 9, stuff: "wrong", created_at: createdAt },
    },
    {
      pageContent: "hi",
      metadata: { b: 2, c: 8, stuff: "right", created_at: createdAt },
    },
    {
      pageContent: "bye",
      metadata: { b: 3, c: 7, stuff: "right", created_at: createdAt },
    },
    {
      pageContent: "what's this",
      metadata: { b: 4, c: 6, stuff: "right", created_at: createdAt },
    },
  ];

  await pgVS.addDocuments(docs, {
    extraColumns: [
      { extra_stuff: "hello 1" },
      { extra_stuff: "hello 2" },
      { extra_stuff: "hello 3" },
      { extra_stuff: "hello 4" },
      { extra_stuff: "hello 5" },
      { extra_stuff: "hello 6" },
      { extra_stuff: "hello 7" },
    ],
  });

  const ssResults = await pgVS.similaritySearch("hello", 2, {
    metadataFilter: {
      b: { $eq: 1 },
    },
  });

  expect(ssResults.length).toBe(2);

  const mmrResults = await pgVS.maxMarginalRelevanceSearch("hello", {
    k: 2,
    fetchK: 7,
    filter: {
      metadataFilter: {
        b: { $eq: 1 },
      },
    },
  });

  expect(mmrResults.length).toBe(2);
});

test("MMR and Similarity Search with filter + join Test pgvector", async () => {
  const embedding = new OpenAIEmbeddings();
  // create some_other_stuff table with pg-promise
  await pgvsPgvector.none(
    "CREATE TABLE IF NOT EXISTS some_extra_stuff (id serial PRIMARY KEY, type varchar(16))"
  );
  // add some data to some_other_stuff table.
  await pgvsPgvector.none(
    "INSERT INTO some_extra_stuff (type) VALUES ('hello'), ('hi'), ('bye')"
  );

  try {
    const pgVS = new PGVectorStore(embedding, {
      postgresConnectionOptions: pgvsPgvector,
      useHnswIndex: true,
      tableName: "pg_embeddings_test_join",
      columns: {
        contentColumnName: "content",
      },
      extraColumns: [
        {
          name: "extra_stuff",
          type: "integer",
          returned: true,
          references: { table: "some_extra_stuff", column: "id" },
        },
      ],
      pgExtensionOpts: { type: "pgvector", dims: 1536 },
    });

    await pgVS.ensureTableInDatabase();

    const createdAt = new Date().getTime();

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
      {
        pageContent: "hello",
        metadata: { b: 1, c: 9, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "hello",
        metadata: { b: 1, c: 9, stuff: "wrong", created_at: createdAt },
      },
      {
        pageContent: "hi",
        metadata: { b: 2, c: 8, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "bye",
        metadata: { b: 3, c: 7, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "what's this",
        metadata: { b: 4, c: 6, stuff: "right", created_at: createdAt },
      },
    ];

    await pgVS.addDocuments(docs, {
      extraColumns: [
        { extra_stuff: 1 },
        { extra_stuff: 1 },
        { extra_stuff: 1 },
        { extra_stuff: 2 },
        { extra_stuff: 2 },
        { extra_stuff: 3 },
        { extra_stuff: 3 },
      ],
    });

    const ssResults = await pgVS.similaritySearch("hello", 3, {
      join: {
        op: "JOIN",
        table: "some_extra_stuff",
        on: [
          {
            left: "pg_embeddings_test_join.extra_stuff",
            right: "some_extra_stuff.id",
            operator: "=",
          },
        ],
      },
      columnFilter: {
        "some_extra_stuff.type": { $eq: "hello" },
      },
    });

    expect(ssResults.length).toBe(3);

    const mmrResults = await pgVS.maxMarginalRelevanceSearch("hello", {
      k: 3,
      fetchK: 7,
      filter: {
        join: {
          op: "JOIN",
          table: "some_extra_stuff",
          on: [
            {
              left: "pg_embeddings_test_join.extra_stuff",
              right: "some_extra_stuff.id",
              operator: "=",
            },
          ],
        },
        columnFilter: {
          "some_extra_stuff.type": { $eq: "hello" },
        },
      },
    });

    expect(mmrResults.length).toBe(3);
  } finally {
    // drop pg_embeddings_test_join table
    await pgvsPgvector.none("DROP TABLE IF EXISTS pg_embeddings_test_join;");
    // drop some extra_stuff table
    await pgvsPgvector.none("DROP TABLE IF EXISTS some_extra_stuff;");
  }
});

test("Text search test pgvector", async () => {
  const embedding = new OpenAIEmbeddings();
  const pgVS = new PGVectorStore(embedding, {
    postgresConnectionOptions: pgvsPgvector,
    useHnswIndex: true,
    tableName: "pg_embeddings",
    columns: {
      contentColumnName: "content",
    },
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtensionOpts: { type: "pgvector", dims: 1536 },
  });

  const createdAt = new Date().getTime();

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
    {
      pageContent: "hello",
      metadata: { b: 1, c: 9, stuff: "right", created_at: createdAt },
    },
    {
      pageContent: "hello",
      metadata: { b: 1, c: 9, stuff: "wrong", created_at: createdAt },
    },
    {
      pageContent: "hi",
      metadata: { b: 2, c: 8, stuff: "right", created_at: createdAt },
    },
    {
      pageContent: "bye",
      metadata: { b: 3, c: 7, stuff: "right", created_at: createdAt },
    },
    {
      pageContent: "what's this",
      metadata: { b: 4, c: 6, stuff: "right", created_at: createdAt },
    },
  ];

  await pgVS.addDocuments(docs, {
    extraColumns: [
      { extra_stuff: "hello 1" },
      { extra_stuff: "hello 2" },
      { extra_stuff: "hello 3" },
      { extra_stuff: "hello 4" },
      { extra_stuff: "hello 5" },
      { extra_stuff: "hello 6" },
      { extra_stuff: "hello 7" },
    ],
  });

  const results = await pgVS.similaritySearch("This is a long text", 1, {
    columnFilter: {
      content: {
        $textSearch: {
          query: `'multidimensional' & 'spaces'`,
          config: "english",
        },
      },
    },
  });

  expect(results.length).toBe(1);
});

test("Build index pgembedding", async () => {
  const embedding = new OpenAIEmbeddings();
  const pgVS = new PGVectorStore(embedding, {
    postgresConnectionOptions: pgvsPgembedding,
    useHnswIndex: true,
    tableName: "pg_embeddings",
    columns: {
      contentColumnName: "content",
    },
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtensionOpts: { type: "pgembedding", dims: 1536 },
  });

  try {
    await pgVS.buildIndex("test_hnsw_index");
    expect(true).toBe(true);
  } catch (error) {
    expect(error).toBe(null);
  } finally {
    await pgVS.dropIndex("test_hnsw_index");
  }
});

test("MMR and Similarity Search Test pgembedding", async () => {
  const embedding = new OpenAIEmbeddings();
  const pgVS = new PGVectorStore(embedding, {
    postgresConnectionOptions: pgvsPgembedding,
    useHnswIndex: true,
    tableName: "pg_embeddings",
    columns: {
      contentColumnName: "content",
    },
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtensionOpts: { type: "pgembedding", dims: 1536 },
  });

  const createdAt = new Date().getTime();

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
    {
      pageContent: "hello",
      metadata: { b: 1, c: 9, stuff: "right", created_at: createdAt },
    },
    {
      pageContent: "hello",
      metadata: { b: 1, c: 9, stuff: "wrong", created_at: createdAt },
    },
    {
      pageContent: "hi",
      metadata: { b: 2, c: 8, stuff: "right", created_at: createdAt },
    },
    {
      pageContent: "bye",
      metadata: { b: 3, c: 7, stuff: "right", created_at: createdAt },
    },
    {
      pageContent: "what's this",
      metadata: { b: 4, c: 6, stuff: "right", created_at: createdAt },
    },
  ];

  await pgVS.addDocuments(docs, {
    extraColumns: [
      { extra_stuff: "hello 1" },
      { extra_stuff: "hello 2" },
      { extra_stuff: "hello 3" },
      { extra_stuff: "hello 4" },
      { extra_stuff: "hello 5" },
      { extra_stuff: "hello 6" },
      { extra_stuff: "hello 7" },
    ],
  });

  const ssResults = await pgVS.similaritySearch("hello", 7);

  expect(ssResults.length).toBe(7);

  const mmrResults = await pgVS.maxMarginalRelevanceSearch("hello", {
    k: 3,
    fetchK: 7,
  });

  expect(mmrResults.length).toBe(3);
});

test("Building WHERE query test pgembedding", () => {
  const embedding = new OpenAIEmbeddings();
  const pgVS = new PGVectorStore(embedding, {
    postgresConnectionOptions: pgvsPgembedding,
    useHnswIndex: true,
    tableName: "pg_embeddings",
    columns: {
      contentColumnName: "content",
    },
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtensionOpts: { type: "pgembedding", dims: 1536 },
  });

  const queryMetadata = pgVS.buildSqlFilterStr(
    {
      $or: [
        { stuff: { $eq: "hello" } },
        { hello: "stuff" },
        {
          $and: [
            { hello: "stuff" },
            {
              content: {
                $textSearch: {
                  query: "hello",
                  config: "english",
                  type: "plain",
                },
              },
            },
          ],
        },
      ],
    },
    "metadata"
  );

  expect(JSON.stringify(queryMetadata)).toBe(
    '["WHERE",{"query":"((($1:raw)::text = $2 OR ($3:raw)::text = $4 OR (($5:raw)::text = $6 AND to_tsvector($7, $8:raw) @@ plainto_tsquery($9, $10))))","values":["metadata->>\'stuff\'","hello","metadata->>\'hello\'","stuff","metadata->>\'hello\'","stuff","english","content","english","hello"]}]'
  );

  const queryColumn = pgVS.buildSqlFilterStr(
    {
      $or: [
        { stuff: { $eq: "hello" } },
        { hello: "stuff" },
        {
          $and: [
            { hello: "stuff" },
            {
              content: {
                $textSearch: {
                  query: "hello",
                  config: "english",
                  type: "plain",
                },
              },
            },
          ],
        },
      ],
    },
    "column"
  );

  expect(JSON.stringify(queryColumn)).toBe(
    '["WHERE",{"query":"((($1:raw) = $2 OR ($3:raw) = $4 OR (($5:raw) = $6 AND to_tsvector($7, $8:raw) @@ plainto_tsquery($9, $10))))","values":["stuff","hello","hello","stuff","hello","stuff","english","content","english","hello"]}]'
  );
});

test("MMR and Similarity Search with filter Test pgembedding", async () => {
  const embedding = new OpenAIEmbeddings();
  const pgVS = new PGVectorStore(embedding, {
    postgresConnectionOptions: pgvsPgembedding,
    useHnswIndex: true,
    tableName: "pg_embeddings",
    columns: {
      contentColumnName: "content",
    },
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtensionOpts: { type: "pgembedding", dims: 1536 },
  });

  const createdAt = new Date().getTime();

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
    {
      pageContent: "hello",
      metadata: { b: 1, c: 9, stuff: "right", created_at: createdAt },
    },
    {
      pageContent: "hello",
      metadata: { b: 1, c: 9, stuff: "wrong", created_at: createdAt },
    },
    {
      pageContent: "hi",
      metadata: { b: 2, c: 8, stuff: "right", created_at: createdAt },
    },
    {
      pageContent: "bye",
      metadata: { b: 3, c: 7, stuff: "right", created_at: createdAt },
    },
    {
      pageContent: "what's this",
      metadata: { b: 4, c: 6, stuff: "right", created_at: createdAt },
    },
  ];

  await pgVS.addDocuments(docs, {
    extraColumns: [
      { extra_stuff: "hello 1" },
      { extra_stuff: "hello 2" },
      { extra_stuff: "hello 3" },
      { extra_stuff: "hello 4" },
      { extra_stuff: "hello 5" },
      { extra_stuff: "hello 6" },
      { extra_stuff: "hello 7" },
    ],
  });

  const ssResults = await pgVS.similaritySearch("hello", 2, {
    metadataFilter: {
      b: { $eq: 1 },
    },
  });

  expect(ssResults.length).toBe(2);

  const mmrResults = await pgVS.maxMarginalRelevanceSearch("hello", {
    k: 2,
    fetchK: 7,
    filter: {
      metadataFilter: {
        b: { $eq: 1 },
      },
    },
  });

  expect(mmrResults.length).toBe(2);
});

test("MMR and Similarity Search with filter + join Test pgembedding", async () => {
  const embedding = new OpenAIEmbeddings();
  // create some_other_stuff table with pg-promise
  await pgvsPgembedding.none(
    "CREATE TABLE IF NOT EXISTS some_extra_stuff (id serial PRIMARY KEY, type varchar(16))"
  );
  // add some data to some_other_stuff table.
  await pgvsPgembedding.none(
    "INSERT INTO some_extra_stuff (type) VALUES ('hello'), ('hi'), ('bye')"
  );

  try {
    const pgVS = new PGVectorStore(embedding, {
      postgresConnectionOptions: pgvsPgembedding,
      useHnswIndex: true,
      tableName: "pg_embeddings_test_join",
      columns: {
        contentColumnName: "content",
      },
      extraColumns: [
        {
          name: "extra_stuff",
          type: "integer",
          returned: true,
          references: { table: "some_extra_stuff", column: "id" },
        },
      ],
      pgExtensionOpts: { type: "pgembedding", dims: 1536 },
    });

    await pgVS.ensureTableInDatabase();

    const createdAt = new Date().getTime();

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
      {
        pageContent: "hello",
        metadata: { b: 1, c: 9, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "hello",
        metadata: { b: 1, c: 9, stuff: "wrong", created_at: createdAt },
      },
      {
        pageContent: "hi",
        metadata: { b: 2, c: 8, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "bye",
        metadata: { b: 3, c: 7, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "what's this",
        metadata: { b: 4, c: 6, stuff: "right", created_at: createdAt },
      },
    ];

    await pgVS.addDocuments(docs, {
      extraColumns: [
        { extra_stuff: 1 },
        { extra_stuff: 1 },
        { extra_stuff: 1 },
        { extra_stuff: 2 },
        { extra_stuff: 2 },
        { extra_stuff: 3 },
        { extra_stuff: 3 },
      ],
    });

    const ssResults = await pgVS.similaritySearch("hello", 3, {
      join: {
        op: "JOIN",
        table: "some_extra_stuff",
        on: [
          {
            left: "pg_embeddings_test_join.extra_stuff",
            right: "some_extra_stuff.id",
            operator: "=",
          },
        ],
      },
      columnFilter: {
        "some_extra_stuff.type": { $eq: "hello" },
      },
    });

    expect(ssResults.length).toBe(3);

    const mmrResults = await pgVS.maxMarginalRelevanceSearch("hello", {
      k: 3,
      fetchK: 7,
      filter: {
        join: {
          op: "JOIN",
          table: "some_extra_stuff",
          on: [
            {
              left: "pg_embeddings_test_join.extra_stuff",
              right: "some_extra_stuff.id",
              operator: "=",
            },
          ],
        },
        columnFilter: {
          "some_extra_stuff.type": { $eq: "hello" },
        },
      },
    });

    expect(mmrResults.length).toBe(3);
  } finally {
    // drop pg_embeddings_test_join table
    await pgvsPgvector.none("DROP TABLE IF EXISTS pg_embeddings_test_join;");
    // drop some extra_stuff table
    await pgvsPgvector.none("DROP TABLE IF EXISTS some_extra_stuff;");
  }
});

test("Text search test pgembedding", async () => {
  const embedding = new OpenAIEmbeddings();
  const pgVS = new PGVectorStore(embedding, {
    postgresConnectionOptions: pgvsPgembedding,
    useHnswIndex: true,
    tableName: "pg_embeddings",
    columns: {
      contentColumnName: "content",
    },
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtensionOpts: { type: "pgembedding", dims: 1536 },
  });

  const createdAt = new Date().getTime();

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
    {
      pageContent: "hello",
      metadata: { b: 1, c: 9, stuff: "right", created_at: createdAt },
    },
    {
      pageContent: "hello",
      metadata: { b: 1, c: 9, stuff: "wrong", created_at: createdAt },
    },
    {
      pageContent: "hi",
      metadata: { b: 2, c: 8, stuff: "right", created_at: createdAt },
    },
    {
      pageContent: "bye",
      metadata: { b: 3, c: 7, stuff: "right", created_at: createdAt },
    },
    {
      pageContent: "what's this",
      metadata: { b: 4, c: 6, stuff: "right", created_at: createdAt },
    },
  ];

  await pgVS.addDocuments(docs, {
    extraColumns: [
      { extra_stuff: "hello 1" },
      { extra_stuff: "hello 2" },
      { extra_stuff: "hello 3" },
      { extra_stuff: "hello 4" },
      { extra_stuff: "hello 5" },
      { extra_stuff: "hello 6" },
      { extra_stuff: "hello 7" },
    ],
  });

  const results = await pgVS.similaritySearch("This is a long text", 1, {
    columnFilter: {
      content: {
        $textSearch: {
          query: `'multidimensional' & 'spaces'`,
          config: "english",
        },
      },
    },
  });

  expect(results.length).toBe(1);
});
