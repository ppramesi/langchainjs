/* eslint-disable no-process-env */
import {
  test,
  expect,
  beforeAll,
  beforeEach,
  afterAll,
  afterEach,
} from "@jest/globals";
import Knex from "knex";
import { OpenAIEmbeddings } from "../../embeddings/openai.js";
import { KnexVectorStore } from "../knex.js";

/**
 * We're using two different postgres instances for each extension. Should setup with docker,
 * see https://github.com/ppramesi/vector-pg-tests/blob/main/database_pgvector/Dockerfile
 * and https://github.com/ppramesi/vector-pg-tests/blob/main/database_pgembedding/Dockerfile
 * for dockerfile configs.
 */
let pgvectorKnex: Knex.Knex;
let pgembeddingKnex: Knex.Knex;

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
  pgvectorKnex = Knex.default({
    client: "postgresql",
    connection: {
      host: process.env.POSTGRES_HOST,
      database: process.env.POSTGRES_PGVECTOR_DB,
      user: process.env.POSTGRES_PGVECTOR_USER,
      password: process.env.POSTGRES_PGVECTOR_PASSWORD,
      port: Number(process.env.POSTGRES_PGVECTOR_PORT),
    },
    pool: { min: 2, max: 20 },
  });
  pgembeddingKnex = Knex.default({
    client: "postgresql",
    connection: {
      host: process.env.POSTGRES_HOST,
      database: process.env.POSTGRES_PGEMBEDDING_DB,
      user: process.env.POSTGRES_PGEMBEDDING_USER,
      password: process.env.POSTGRES_PGEMBEDDING_PASSWORD,
      port: Number(process.env.POSTGRES_PGEMBEDDING_PORT),
    },
    pool: { min: 2, max: 20 },
  });
});

beforeEach(async () => {
  /**
   * 🚨🚨🚨 WARNING WARNING WARNING 🚨🚨🚨
   * We're dropping knex_embeddings table first to make sure the test
   * is idempotent. This means that if you have a table called
   * knex_embeddings in your database, it will be dropped.
   */
  await Promise.all([
    pgvectorKnex.schema.dropTableIfExists("knex_embeddings"),
    pgembeddingKnex.schema.dropTableIfExists("knex_embeddings"),
  ]);

  const embedding = new OpenAIEmbeddings();

  const pgvKnexVS = new KnexVectorStore(embedding, {
    knex: pgvectorKnex,
    useHnswIndex: false,
    tableName: "knex_embeddings",
    pageContentColumn: "content",
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtension: "pgvector",
  });
  const pgeKnexVS = new KnexVectorStore(embedding, {
    knex: pgembeddingKnex,
    useHnswIndex: false,
    tableName: "knex_embeddings",
    pageContentColumn: "content",
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtension: "pgembedding",
  });

  await Promise.all([
    pgvKnexVS.ensureTableInDatabase(),
    pgeKnexVS.ensureTableInDatabase(),
  ]);
});

afterEach(async () => {
  await Promise.all([
    pgvectorKnex.schema.dropTableIfExists("knex_embeddings"),
    pgembeddingKnex.schema.dropTableIfExists("knex_embeddings"),
  ]);
});

afterAll(async () => {
  await Promise.all([pgvectorKnex.destroy(), pgembeddingKnex.destroy()]);
});

test("Build index pgvector", async () => {
  const embedding = new OpenAIEmbeddings();
  const knexVS = new KnexVectorStore(embedding, {
    knex: pgvectorKnex,
    useHnswIndex: false,
    tableName: "knex_embeddings",
    pageContentColumn: "content",
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtension: "pgvector",
  });

  try {
    await knexVS.buildIndex("test_hnsw_index");
    expect(true).toBe(true);
  } catch (error) {
    expect(error).toBe(null);
  } finally {
    await knexVS.dropIndex("test_hnsw_index");
  }
});

test("MMR and Similarity Search Test pgvector", async () => {
  const embedding = new OpenAIEmbeddings();
  const knexVS = new KnexVectorStore(embedding, {
    knex: pgvectorKnex,
    useHnswIndex: false,
    tableName: "knex_embeddings",
    pageContentColumn: "content",
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtension: "pgvector",
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

  await knexVS.addDocuments(docs, {
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

  const ssResults = await knexVS.similaritySearch("hello", 7);

  expect(ssResults.length).toBe(7);

  const mmrResults = await knexVS.maxMarginalRelevanceSearch("hello", {
    k: 3,
    fetchK: 7,
  });

  expect(mmrResults.length).toBe(3);
});

test("Building WHERE query test pgvector", () => {
  const embedding = new OpenAIEmbeddings();
  const knexVS = new KnexVectorStore(embedding, {
    knex: pgvectorKnex,
    useHnswIndex: false,
    tableName: "knex_embeddings",
    pageContentColumn: "content",
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtension: "pgvector",
  });

  const queryMetadata = knexVS.buildSqlFilterStr(
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

  expect(queryMetadata).toBe(
    `WHERE (metadata->>'stuff' = 'hello'::text OR metadata->>'hello' = 'stuff'::text OR (metadata->>'hello' = 'stuff'::text AND to_tsvector('english', 'content') @@ plainto_tsquery('english', 'hello')))`
  );

  const queryColumn = knexVS.buildSqlFilterStr(
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

  expect(queryColumn).toBe(
    `WHERE (stuff = 'hello' OR hello = 'stuff' OR (hello = 'stuff' AND to_tsvector('english', 'content') @@ plainto_tsquery('english', 'hello')))`
  );
});

test("Build index pgembedding", async () => {
  const embedding = new OpenAIEmbeddings();
  const knexVS = new KnexVectorStore(embedding, {
    knex: pgembeddingKnex,
    useHnswIndex: false,
    tableName: "knex_embeddings",
    pageContentColumn: "content",
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtension: "pgembedding",
  });

  try {
    await knexVS.buildIndex("test_hnsw_index");
    expect(true).toBe(true);
  } catch (error) {
    expect(error).toBe(null);
  } finally {
    await knexVS.dropIndex("test_hnsw_index");
  }
});

test("MMR and Similarity Search Test pgembedding", async () => {
  const embedding = new OpenAIEmbeddings();
  const knexVS = new KnexVectorStore(embedding, {
    knex: pgembeddingKnex,
    useHnswIndex: false,
    tableName: "knex_embeddings",
    pageContentColumn: "content",
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtension: "pgembedding",
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

  await knexVS.addDocuments(docs, {
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

  const ssResults = await knexVS.similaritySearch("hello", 7);

  expect(ssResults.length).toBe(7);

  const mmrResults = await knexVS.maxMarginalRelevanceSearch("hello", {
    k: 3,
    fetchK: 7,
  });

  expect(mmrResults.length).toBe(3);
});

test("Building WHERE query test pgembedding", () => {
  const embedding = new OpenAIEmbeddings();
  const knexVS = new KnexVectorStore(embedding, {
    knex: pgembeddingKnex,
    useHnswIndex: false,
    tableName: "knex_embeddings",
    pageContentColumn: "content",
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtension: "pgembedding",
  });

  const queryMetadata = knexVS.buildSqlFilterStr(
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

  expect(queryMetadata).toBe(
    `WHERE (metadata->>'stuff' = 'hello'::text OR metadata->>'hello' = 'stuff'::text OR (metadata->>'hello' = 'stuff'::text AND to_tsvector('english', 'content') @@ plainto_tsquery('english', 'hello')))`
  );

  const queryColumn = knexVS.buildSqlFilterStr(
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

  expect(queryColumn).toBe(
    `WHERE (stuff = 'hello' OR hello = 'stuff' OR (hello = 'stuff' AND to_tsvector('english', 'content') @@ plainto_tsquery('english', 'hello')))`
  );
});
